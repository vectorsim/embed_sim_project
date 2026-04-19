/**********************************************************************************************************************
 * \file        cdd_motor_command_queue.c
 * \brief       Self-contained implementation of cdd_motor_command_queue.h —
 *              bidirectional MPSC priority queue for Core0 ↔ Core1 IPC on
 *              AURIX TC3xx.
 *
 * \details     All queue logic (atomic primitives, priority routing, ring
 *              management, field-copy helper) is implemented directly here.
 *              No dependency on a separate cdd_ipc_queue module exists.
 *
 *              Internal structure (same for both channels):
 *
 *                  MCQ_Copy_Msg()        — field-by-field copy (Rule 21.6)
 *                  MCQ_Resolve_Priority()— elevate MOTOR_STOP to HIGH
 *                  MCQ_Enqueue_High_()   — CMPSWAP.W claim of high slot
 *                  MCQ_Enqueue_Normal_() — CMPSWAP.W fetch-and-increment ring
 *                  MCQ_Enqueue_()        — routes to High or Normal path
 *                  MCQ_Dequeue_()        — drains High first, then Normal
 *
 *              Public functions delegate to these helpers with the appropriate
 *              MCQ_Queue_T pointer (MCQ_Cmd_Queue_G or MCQ_Stat_Queue_G).
 *
 *              Atomic correctness (TC3xx LMU):
 *                  High slot  : CMPSWAP.W ensures only one producer writes the
 *                               slot; the slot is already marked OCCUPIED before
 *                               the consumer can observe it.
 *                  Normal ring: CMPSWAP.W fetch-and-increment guarantees each
 *                               producer gets a unique slot index.  Slot content
 *                               is written after index reservation; wr_idx is
 *                               advanced by the CMPSWAP itself so the consumer
 *                               sees both the index and the content in the
 *                               correct order on the coherent LMU bus.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_motor_command_queue.h"

/**********************************************************************************************************************
 * Queue Control Blocks — LMU RAM (coherent across all TC3xx cores)
 *********************************************************************************************************************/

/** \brief Command channel: Core0 → Core1.  */
MCQ_Queue_T MCQ_Cmd_Queue_G  __attribute__((section(".lmu_data")));

/** \brief Status channel:  Core1 → Core0.  */
MCQ_Queue_T MCQ_Stat_Queue_G __attribute__((section(".lmu_data")));

/**********************************************************************************************************************
 * Sequence Counters
 *********************************************************************************************************************/

/** \brief Monotonic counter for command messages (Core0-owned).               */
volatile uint32_T MCQ_Cmd_Seq_G  __attribute__((section(".lmu_data"))) = 0U;

/** \brief Monotonic counter for status messages (Core1-owned).                */
volatile uint32_T MCQ_Stat_Seq_G __attribute__((section(".lmu_data"))) = 0U;

/**********************************************************************************************************************
 * Private Helper — field-by-field copy  (MISRA Rule 21.6: no memcpy)
 *********************************************************************************************************************/

/**
 * \brief   Copies one MCQ_Msg_T by field assignment.
 * \param   Dst   Destination message buffer
 * \param   Src   Source message buffer
 */
static void MCQ_Copy_Msg(MCQ_Msg_T * const Dst, const MCQ_Msg_T * const Src)
{
    Dst->msg_id  = Src->msg_id;
    Dst->seq_num = Src->seq_num;
    Dst->data[0] = Src->data[0];
    Dst->data[1] = Src->data[1];
}

/**********************************************************************************************************************
 * Private Helper — resolve effective priority
 *********************************************************************************************************************/

/**
 * \brief   Returns MCQ_PRIO_HIGH if msg_id is MOTOR_STOP, else the supplied
 *          priority.
 * \param   msg_id    Message identifier
 * \param   priority  Caller-supplied priority
 * \return  Effective MCQ_Priority_T
 */
static MCQ_Priority_T MCQ_Resolve_Priority(uint32_T       msg_id,
                                            MCQ_Priority_T priority)
{
    MCQ_Priority_T effective;

    if (msg_id == (uint32_T)MCQ_MSG_MOTOR_STOP)
    {
        effective = MCQ_PRIO_HIGH;
    }
    else
    {
        effective = priority;
    }

    return effective;
}

/**********************************************************************************************************************
 * Private Helper — high slot enqueue via CMPSWAP
 *********************************************************************************************************************/

/**
 * \brief   Attempts to write one message into the high-priority slot.
 *
 * \details Uses CMPSWAP.W to atomically claim the slot.  If already occupied
 *          the message is dropped and MCQ_HIGH_BUSY is returned — the producer
 *          must retry after the consumer drains the slot.
 *
 * \param   Queue_Ptr   Pointer to the target queue
 * \param   Msg_Ptr     Message to write
 * \return  MCQ_OK | MCQ_HIGH_BUSY
 */
static MCQ_Status_T MCQ_Enqueue_High_(MCQ_Queue_T     * const Queue_Ptr,
                                       const MCQ_Msg_T * const Msg_Ptr)
{
    MCQ_Status_T status;
    uint32_T     prev;

    /* Atomically claim: if high_valid == FREE write OCCUPIED, return previous  */
    prev = ASM_Cmp_And_Swap((uint32_T *)&Queue_Ptr->high_valid,
                             MCQ_HIGH_SLOT_OCCUPIED,
                             MCQ_HIGH_SLOT_FREE);

    if (prev == MCQ_HIGH_SLOT_FREE)
    {
        /* Claim succeeded — write payload.  high_valid is already OCCUPIED.
         * The consumer sees OCCUPIED and then reads the slot; both accesses
         * are on the coherent LMU bus so no explicit fence is required.       */
        MCQ_Copy_Msg(&Queue_Ptr->high_slot, Msg_Ptr);
        status = MCQ_OK;
    }
    else
    {
        /* Slot already held — caller must retry after consumer drains it       */
        status = MCQ_HIGH_BUSY;
    }

    return status;
}

/**********************************************************************************************************************
 * Private Helper — normal ring enqueue via CMPSWAP fetch-and-increment
 *********************************************************************************************************************/

/**
 * \brief   Reserves a slot in the normal ring and writes the message.
 *
 * \details Performs a lock-free fetch-and-increment on wr_idx using a
 *          CMPSWAP.W retry loop.  With at most two producers the loop
 *          completes in at most two iterations.
 *
 * \param   Queue_Ptr   Pointer to the target queue
 * \param   Msg_Ptr     Message to write
 * \return  MCQ_OK | MCQ_FULL
 */
static MCQ_Status_T MCQ_Enqueue_Normal_(MCQ_Queue_T     * const Queue_Ptr,
                                         const MCQ_Msg_T * const Msg_Ptr)
{
    MCQ_Status_T status;
    uint32_T     current_wr;
    uint32_T     next_wr;
    uint32_T     prev;
    uint32_T     occupancy;
    uint32_T     slot_idx;

    do
    {
        current_wr = Queue_Ptr->wr_idx;
        occupancy  = current_wr - Queue_Ptr->rd_idx;

        if (occupancy >= MCQ_QUEUE_NORM_DEPTH)
        {
            return MCQ_FULL;   /* Ring is full — no slot to claim               */
        }

        next_wr = current_wr + 1U;

        /* Attempt to reserve: if wr_idx == current_wr, advance to next_wr     */
        prev = ASM_Cmp_And_Swap((uint32_T *)&Queue_Ptr->wr_idx,
                                 next_wr,
                                 current_wr);

    } while (prev != current_wr);   /* Retry if another producer got there first */

    /* Exclusive slot ownership confirmed — write the message payload           */
    slot_idx = current_wr & MCQ_QUEUE_NORM_MASK;
    MCQ_Copy_Msg(&Queue_Ptr->slots[slot_idx], Msg_Ptr);

    /* wr_idx was already advanced by the CMPSWAP — consumer can now see slot  */
    status = MCQ_OK;

    return status;
}

/**********************************************************************************************************************
 * Private Helper — unified enqueue (routes to High or Normal path)
 *********************************************************************************************************************/

/**
 * \brief   Enqueues one message into the given queue with priority routing.
 *
 * \param   Queue_Ptr   Pointer to the target queue
 * \param   Msg_Ptr     Message to enqueue
 * \param   Priority    Requested priority (MOTOR_STOP always overrides to HIGH)
 * \return  MCQ_OK | MCQ_FULL | MCQ_HIGH_BUSY
 */
static MCQ_Status_T MCQ_Enqueue_(MCQ_Queue_T     * const Queue_Ptr,
                                  const MCQ_Msg_T * const Msg_Ptr,
                                  MCQ_Priority_T          Priority)
{
    MCQ_Priority_T effective;
    MCQ_Status_T   status;

    effective = MCQ_Resolve_Priority(Msg_Ptr->msg_id, Priority);

    if (effective == MCQ_PRIO_HIGH)
    {
        status = MCQ_Enqueue_High_(Queue_Ptr, Msg_Ptr);
    }
    else
    {
        status = MCQ_Enqueue_Normal_(Queue_Ptr, Msg_Ptr);
    }

    return status;
}

/**********************************************************************************************************************
 * Private Helper — unified dequeue (drains High first, then Normal)
 *********************************************************************************************************************/

/**
 * \brief   Dequeues one message from the given queue (single consumer only).
 *
 * \param   Queue_Ptr   Pointer to the target queue
 * \param   Msg_Ptr     Output buffer for the dequeued message
 * \return  MCQ_OK | MCQ_EMPTY
 */
static MCQ_Status_T MCQ_Dequeue_(MCQ_Queue_T * const Queue_Ptr,
                                  MCQ_Msg_T   * const Msg_Ptr)
{
    MCQ_Status_T status;
    uint32_T     slot_idx;

    /* --- High-priority slot — drain first --- */
    if (Queue_Ptr->high_valid == MCQ_HIGH_SLOT_OCCUPIED)
    {
        MCQ_Copy_Msg(Msg_Ptr, &Queue_Ptr->high_slot);

        /* Release slot — volatile write, immediately visible to producers      */
        Queue_Ptr->high_valid = MCQ_HIGH_SLOT_FREE;

        status = MCQ_OK;
    }
    /* --- Normal ring --- */
    else if (Queue_Ptr->wr_idx != Queue_Ptr->rd_idx)
    {
        slot_idx = Queue_Ptr->rd_idx & MCQ_QUEUE_NORM_MASK;
        MCQ_Copy_Msg(Msg_Ptr, &Queue_Ptr->slots[slot_idx]);

        /* Advance read index — consumer-owned, no atomic needed                */
        Queue_Ptr->rd_idx = Queue_Ptr->rd_idx + 1U;

        status = MCQ_OK;
    }
    else
    {
        status = MCQ_EMPTY;
    }

    return status;
}

/**********************************************************************************************************************
 * Private Helper — single queue initialisation
 *********************************************************************************************************************/

/**
 * \brief   Resets one MCQ_Queue_T to a clean empty state.
 * \param   Queue_Ptr   Pointer to the queue to initialise
 */
static void MCQ_Init_Queue_(MCQ_Queue_T * const Queue_Ptr)
{
    uint32_T i;

    Queue_Ptr->high_valid        = MCQ_HIGH_SLOT_FREE;
    Queue_Ptr->high_slot.msg_id  = (uint32_T)MCQ_MSG_NONE;
    Queue_Ptr->high_slot.seq_num = 0U;
    Queue_Ptr->high_slot.data[0] = 0.0f;
    Queue_Ptr->high_slot.data[1] = 0.0f;

    Queue_Ptr->wr_idx = 0U;
    Queue_Ptr->rd_idx = 0U;

    for (i = 0U; i < MCQ_QUEUE_NORM_DEPTH; i++)
    {
        Queue_Ptr->slots[i].msg_id  = (uint32_T)MCQ_MSG_NONE;
        Queue_Ptr->slots[i].seq_num = 0U;
        Queue_Ptr->slots[i].data[0] = 0.0f;
        Queue_Ptr->slots[i].data[1] = 0.0f;
    }
}

/**********************************************************************************************************************
 * Public — Initialisation
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Init
 *------------------------------------------------------------------------------------------------------------------*/
void MCQ_Init(void)
{
    MCQ_Init_Queue_(&MCQ_Cmd_Queue_G);
    MCQ_Init_Queue_(&MCQ_Stat_Queue_G);

    MCQ_Cmd_Seq_G  = 0U;
    MCQ_Stat_Seq_G = 0U;
}

/**********************************************************************************************************************
 * Public — Command Channel
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Enqueue_Cmd
 *------------------------------------------------------------------------------------------------------------------*/
MCQ_Status_T MCQ_Enqueue_Cmd(const MCQ_Msg_T * const Msg_Ptr,
                               MCQ_Priority_T          Priority)
{
    return MCQ_Enqueue_(&MCQ_Cmd_Queue_G, Msg_Ptr, Priority);
}

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Dequeue_Cmd
 *------------------------------------------------------------------------------------------------------------------*/
MCQ_Status_T MCQ_Dequeue_Cmd(MCQ_Msg_T * const Msg_Ptr)
{
    return MCQ_Dequeue_(&MCQ_Cmd_Queue_G, Msg_Ptr);
}

/**********************************************************************************************************************
 * Public — Status Channel
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Enqueue_Stat
 *------------------------------------------------------------------------------------------------------------------*/
MCQ_Status_T MCQ_Enqueue_Stat(const MCQ_Msg_T * const Msg_Ptr,
                                MCQ_Priority_T          Priority)
{
    return MCQ_Enqueue_(&MCQ_Stat_Queue_G, Msg_Ptr, Priority);
}

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Dequeue_Stat
 *------------------------------------------------------------------------------------------------------------------*/
MCQ_Status_T MCQ_Dequeue_Stat(MCQ_Msg_T * const Msg_Ptr)
{
    return MCQ_Dequeue_(&MCQ_Stat_Queue_G, Msg_Ptr);
}

/**********************************************************************************************************************
 * Public — Diagnostic Helpers
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Cmd_Stop_Pending
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T MCQ_Cmd_Stop_Pending(void)
{
    return (MCQ_Cmd_Queue_G.high_valid == MCQ_HIGH_SLOT_OCCUPIED) ? 1U : 0U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Cmd_Count
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T MCQ_Cmd_Count(void)
{
    return MCQ_Cmd_Queue_G.wr_idx - MCQ_Cmd_Queue_G.rd_idx;
}

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Stat_Count
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T MCQ_Stat_Count(void)
{
    return MCQ_Stat_Queue_G.wr_idx - MCQ_Stat_Queue_G.rd_idx;
}

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Cmd_Is_Empty
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T MCQ_Cmd_Is_Empty(void)
{
    uint32_T high_empty;
    uint32_T norm_empty;

    high_empty = (MCQ_Cmd_Queue_G.high_valid == MCQ_HIGH_SLOT_FREE) ? 1U : 0U;
    norm_empty = (MCQ_Cmd_Queue_G.wr_idx     == MCQ_Cmd_Queue_G.rd_idx) ? 1U : 0U;

    return ((high_empty == 1U) && (norm_empty == 1U)) ? 1U : 0U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * MCQ_Cmd_Is_Full
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T MCQ_Cmd_Is_Full(void)
{
    uint32_T occupancy;

    occupancy = MCQ_Cmd_Queue_G.wr_idx - MCQ_Cmd_Queue_G.rd_idx;

    return (occupancy >= MCQ_QUEUE_NORM_DEPTH) ? 1U : 0U;
}
