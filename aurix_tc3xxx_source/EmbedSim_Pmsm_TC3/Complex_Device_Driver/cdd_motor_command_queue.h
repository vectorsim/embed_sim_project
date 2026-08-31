/**********************************************************************************************************************
 * \file        cdd_motor_command_queue.h
 * \brief       Self-contained lockless MPSC bidirectional IPC channel between
 *              Core0 (CAN gateway) and Core1 (motor FOC controller) on
 *              AURIX TC3xx.
 *
 * \details     Physical wiring (two independent MPSC queues in LMU RAM):
 *
 *                  Core0 (CAN IRQ)  ──►  MCQ_Cmd_Queue_G   ──►  Core1 (FOC)
 *                  Core1 (FOC ISR)  ──►  MCQ_Stat_Queue_G  ──►  Core0 (CAN TX)
 *
 *              Architecture — two logical sub-queues per channel:
 *
 *                  [HIGH]  Priority slot — depth 1, reserved for MOTOR_STOP and
 *                          any message sent with MCQ_PRIO_HIGH.  Claimed
 *                          atomically via CMPSWAP.W so two producers racing for
 *                          it are safe.
 *
 *                  [NORM]  Normal ring-buffer — depth MCQ_QUEUE_NORM_DEPTH (8).
 *                          wr_idx reservation is atomic via CMPSWAP.W retry
 *                          loop so CPU0 and CPU1 can enqueue concurrently.
 *
 *              Atomic primitives used (from cdd_asm_functions):
 *
 *              1. High slot claim (CMPSWAP.W):
 *
 *                   ASM_Cmp_And_Swap(&high_valid,
 *                                    MCQ_HIGH_SLOT_OCCUPIED,   // new
 *                                    MCQ_HIGH_SLOT_FREE);      // expected
 *
 *                  Returns previous value: FREE → claim succeeded,
 *                  OCCUPIED → slot busy, caller receives MCQ_HIGH_BUSY.
 *
 *              2. Normal ring slot reservation (CMPSWAP.W retry loop):
 *
 *                   do {
 *                       current = wr_idx;
 *                       next    = current + 1U;
 *                       prev    = CMPSWAP(&wr_idx, next, current);
 *                   } while (prev != current);
 *
 *                  Lock-free fetch-and-increment.  With at most two producers
 *                  the loop completes in at most two iterations.
 *
 *              Memory ordering on TC3xx LMU:
 *                  LMU has no per-core data cache.  A volatile write is visible
 *                  to all cores immediately.  Slot content is always written
 *                  before wr_idx is advanced so the consumer never observes a
 *                  partially written slot.
 *
 *              Dequeue policy (single consumer per channel):
 *                  1. High-priority slot always drained first.
 *                  2. Normal ring drained in FIFO order.
 *
 *              MCQ_MSG_MOTOR_STOP forces MCQ_PRIO_HIGH regardless of the
 *              priority argument supplied by the caller.
 *
 *              Memory placement:
 *                  All instances MUST reside in LMU RAM (section ".lmu_data").
 *
 *              Usage sketch — Core0 CAN receive ISR:
 *
 *                  MCQ_Msg_T msg;
 *                  msg.msg_id  = (uint32_T)MCQ_MSG_MOTOR_CMD;
 *                  msg.seq_num = MCQ_Cmd_Seq_G++;
 *                  msg.data[0] = speed_ref_rpm;
 *                  msg.data[1] = torque_limit_Nm;
 *                  (void)MCQ_Enqueue_Cmd(&msg, MCQ_PRIO_NORMAL);
 *
 *              Usage sketch — Core1 FOC ISR:
 *
 *                  MCQ_Msg_T    msg;
 *                  MCQ_Status_T s = MCQ_Dequeue_Cmd(&msg);
 *                  if (s == MCQ_OK) { ... apply msg ... }
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  6.1  : All bit-field types unsigned
 *              - Rule  8.5  : Each object declared once in this header
 *              - Rule  8.6  : All definitions in cdd_motor_command_queue.c
 *              - Rule 14.4  : All conditions use explicit comparison
 *              - Rule 17.2  : No recursion
 *              - Rule 20.5  : No #undef
 *              - Rule 21.6  : No stdio; field-by-field copy replaces memcpy
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_MOTOR_COMMAND_QUEUE_H_
#define CDD_MOTOR_COMMAND_QUEUE_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"
#include "cdd_asm_functions.h"

/**********************************************************************************************************************
 * Configuration
 *********************************************************************************************************************/

/** \brief  Normal ring-buffer depth — must be a power of two.                 */
#define MCQ_QUEUE_NORM_DEPTH        (8U)

/** \brief  Normal ring index wrap mask.                                        */
#define MCQ_QUEUE_NORM_MASK         (MCQ_QUEUE_NORM_DEPTH - 1U)

/** \brief  high_valid sentinel: slot is occupied.                              */
#define MCQ_HIGH_SLOT_OCCUPIED      (1U)

/** \brief  high_valid sentinel: slot is free.                                  */
#define MCQ_HIGH_SLOT_FREE          (0U)

/** \brief  Compile-time power-of-two guard.                                    */
typedef char MCQ_PowerOfTwo_Check
    [ ((MCQ_QUEUE_NORM_DEPTH & (MCQ_QUEUE_NORM_DEPTH - 1U)) == 0U) ? 1 : -1 ];

/**********************************************************************************************************************
 * Message Priority
 *********************************************************************************************************************/

/**
 * \brief   Message dispatch priority.
 */
typedef enum
{
    MCQ_PRIO_NORMAL = 0U,   /**< \brief Routes to normal FIFO ring              */
    MCQ_PRIO_HIGH   = 1U    /**< \brief Routes to high-priority slot            */
} MCQ_Priority_T;

/**********************************************************************************************************************
 * Message Identifier
 *********************************************************************************************************************/

/**
 * \brief   IPC message type.
 *
 * \note    MCQ_MSG_MOTOR_STOP is always elevated to MCQ_PRIO_HIGH by the
 *          enqueue function, regardless of the caller-supplied priority.
 */
typedef enum
{
    MCQ_MSG_NONE        = 0x00U,    /**< \brief No message / initialised        */
    MCQ_MSG_MOTOR_CMD   = 0x01U,    /**< \brief Motor control setpoint          */
    MCQ_MSG_MOTOR_STAT  = 0x02U,    /**< \brief Motor status feedback           */
    MCQ_MSG_MOTOR_STOP  = 0x03U,    /**< \brief EMERGENCY STOP — always HIGH    */
    MCQ_MSG_FAULT       = 0x04U,    /**< \brief Fault / diagnostic notification */
    MCQ_MSG_USER        = 0xFFU     /**< \brief Application-defined extension   */
} MCQ_Msg_Id_T;

/**********************************************************************************************************************
 * Message Payload  (16 bytes, 4-byte aligned)
 *********************************************************************************************************************/

/**
 * \brief   Fixed-size message payload.
 *
 * \details Total size is a multiple of 4 bytes for aligned LMU word access.
 *          Extend data[] or add fields as required by the application.
 *          CAN frame fields should be marshalled into data[0]/data[1] by
 *          the CAN receive handler before calling MCQ_Enqueue_Cmd.
 */
typedef struct
{
    uint32_T    msg_id;         /**< \brief Message type  (MCQ_Msg_Id_T)        [–] */
    uint32_T    seq_num;        /**< \brief Monotonic sequence counter          [–] */
    real32_T    data[2];        /**< \brief Application payload (two floats)    [–] */
} MCQ_Msg_T;

/**********************************************************************************************************************
 * Queue Control Block
 *
 *  high_valid  : 0 = free, 1 = occupied  — CMPSWAP target for producers
 *  high_slot   : single high-priority message storage
 *  wr_idx      : next claimed write index — CMPSWAP target for producers
 *  rd_idx      : next read index          — consumer-owned, no atomics needed
 *  slots[]     : normal-priority ring buffer
 *********************************************************************************************************************/
typedef struct
{
    volatile uint32_T   high_valid;                         /**< \brief High slot occupancy flag    [–] */
    MCQ_Msg_T           high_slot;                          /**< \brief High-priority message           */
    volatile uint32_T   wr_idx;                             /**< \brief Normal ring write index     [–] */
    volatile uint32_T   rd_idx;                             /**< \brief Normal ring read index      [–] */
    MCQ_Msg_T           slots[MCQ_QUEUE_NORM_DEPTH];        /**< \brief Normal ring buffer              */
} MCQ_Queue_T;

/**********************************************************************************************************************
 * Return Codes
 *********************************************************************************************************************/

typedef enum
{
    MCQ_OK          = 0U,   /**< \brief Operation succeeded                         */
    MCQ_FULL        = 1U,   /**< \brief Normal ring full — message dropped          */
    MCQ_EMPTY       = 2U,   /**< \brief Both queues empty on dequeue                */
    MCQ_HIGH_BUSY   = 3U    /**< \brief High slot occupied — producer must retry    */
} MCQ_Status_T;

/**********************************************************************************************************************
 * Shared Queue Instances  (defined in cdd_motor_command_queue.c, LMU section)
 *********************************************************************************************************************/

/**
 * \brief  Command channel: Core0 (CAN) → Core1 (FOC).
 *         Producers: Core0 (CAN receive ISR / task).
 *         Consumer : Core1 (FOC ISR / scheduler).
 */
extern MCQ_Queue_T MCQ_Cmd_Queue_G;

/**
 * \brief  Status channel: Core1 (FOC) → Core0 (CAN TX / diagnostics).
 *         Producers: Core1 (FOC ISR).
 *         Consumer : Core0 (CAN TX task / diagnostic logger).
 */
extern MCQ_Queue_T MCQ_Stat_Queue_G;

/**********************************************************************************************************************
 * Convenience Sequence Counters
 *********************************************************************************************************************/

/** \brief Monotonic counter for command messages — incremented by Core0.      */
extern volatile uint32_T MCQ_Cmd_Seq_G;

/** \brief Monotonic counter for status messages — incremented by Core1.       */
extern volatile uint32_T MCQ_Stat_Seq_G;

/**********************************************************************************************************************
 * Function Prototypes — Initialisation
 *********************************************************************************************************************/

/**
 * \brief   Initialises both queue control blocks to a clean empty state.
 *
 * \details Must be called once during system start-up before any core begins
 *          enqueuing.  Typically called from Core0 after LMU RAM is accessible.
 *          Both sequence counters are reset to 0.
 *
 * \return  None
 */
extern void MCQ_Init(void);

/**********************************************************************************************************************
 * Function Prototypes — Command Channel  (Core0 producer → Core1 consumer)
 *********************************************************************************************************************/

/**
 * \brief   Enqueues one command message from Core0 to Core1.
 *
 * \details MCQ_MSG_MOTOR_STOP is automatically elevated to MCQ_PRIO_HIGH.
 *          Non-blocking — never busy-waits.
 *
 * \param   Msg_Ptr   Message to enqueue (msg_id and data[] set by caller)
 * \param   Priority  MCQ_PRIO_NORMAL | MCQ_PRIO_HIGH
 * \return  MCQ_OK | MCQ_FULL | MCQ_HIGH_BUSY
 */
extern MCQ_Status_T MCQ_Enqueue_Cmd(const MCQ_Msg_T * const Msg_Ptr,
                                     MCQ_Priority_T          Priority);

/**
 * \brief   Dequeues one command message on Core1 (single consumer).
 *
 * \details High-priority slot (MOTOR_STOP) drained first, then normal ring.
 *          Non-blocking — returns MCQ_EMPTY if nothing is pending.
 *
 * \param   Msg_Ptr   Output buffer for the dequeued message
 * \return  MCQ_OK | MCQ_EMPTY
 */
extern MCQ_Status_T MCQ_Dequeue_Cmd(MCQ_Msg_T * const Msg_Ptr);

/**********************************************************************************************************************
 * Function Prototypes — Status Channel  (Core1 producer → Core0 consumer)
 *********************************************************************************************************************/

/**
 * \brief   Enqueues one status/feedback message from Core1 to Core0.
 *
 * \details MCQ_MSG_FAULT should be sent with MCQ_PRIO_HIGH so it appears at
 *          the head of the status stream.
 *
 * \param   Msg_Ptr   Message to enqueue
 * \param   Priority  MCQ_PRIO_NORMAL | MCQ_PRIO_HIGH
 * \return  MCQ_OK | MCQ_FULL | MCQ_HIGH_BUSY
 */
extern MCQ_Status_T MCQ_Enqueue_Stat(const MCQ_Msg_T * const Msg_Ptr,
                                      MCQ_Priority_T          Priority);

/**
 * \brief   Dequeues one status message on Core0 (single consumer).
 *
 * \param   Msg_Ptr   Output buffer for the dequeued message
 * \return  MCQ_OK | MCQ_EMPTY
 */
extern MCQ_Status_T MCQ_Dequeue_Stat(MCQ_Msg_T * const Msg_Ptr);

/**********************************************************************************************************************
 * Function Prototypes — Diagnostic Helpers
 *********************************************************************************************************************/

/**
 * \brief   Returns 1 if a MOTOR_STOP is pending in the command high slot.
 * \return  1 = STOP pending, 0 = free
 */
extern uint32_T MCQ_Cmd_Stop_Pending(void);

/**
 * \brief   Returns occupancy of the normal command ring [0..MCQ_QUEUE_NORM_DEPTH].
 * \return  Number of normal-priority command messages queued
 */
extern uint32_T MCQ_Cmd_Count(void);

/**
 * \brief   Returns occupancy of the normal status ring [0..MCQ_QUEUE_NORM_DEPTH].
 * \return  Number of normal-priority status messages queued
 */
extern uint32_T MCQ_Stat_Count(void);

/**
 * \brief   Returns 1 if the command channel (high slot + ring) is completely empty.
 * \return  1 = empty, 0 = message(s) pending
 */
extern uint32_T MCQ_Cmd_Is_Empty(void);

/**
 * \brief   Returns 1 if the normal command ring is full.
 * \return  1 = full, 0 = space available
 */
extern uint32_T MCQ_Cmd_Is_Full(void);

#endif /* CDD_MOTOR_COMMAND_QUEUE_H_ */
