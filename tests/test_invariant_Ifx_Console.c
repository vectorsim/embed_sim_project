#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>

/* Include the header for Ifx_Console functions */
#include "Ifx_Console.h"

static sigjmp_buf jump_buffer;
static volatile sig_atomic_t buffer_overflow_detected = 0;

static void segfault_handler(int sig)
{
    buffer_overflow_detected = 1;
    siglongjmp(jump_buffer, 1);
}

START_TEST(test_console_print_buffer_bounds)
{
    /* Invariant: Buffer reads/writes never exceed declared buffer length */
    
    /* Payloads: exploit case (2x buffer), boundary case (10x buffer), valid input */
    char payload_2x[512];   /* 2x typical 256-byte buffer */
    char payload_10x[2560]; /* 10x typical buffer */
    const char *payload_valid = "Short valid message";
    
    memset(payload_2x, 'A', sizeof(payload_2x) - 1);
    payload_2x[sizeof(payload_2x) - 1] = '\0';
    
    memset(payload_10x, 'B', sizeof(payload_10x) - 1);
    payload_10x[sizeof(payload_10x) - 1] = '\0';
    
    const char *payloads[] = {
        payload_valid,
        payload_2x,
        payload_10x
    };
    int num_payloads = sizeof(payloads) / sizeof(payloads[0]);
    
    struct sigaction sa, old_sa;
    sa.sa_handler = segfault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &old_sa);
    
    for (int i = 0; i < num_payloads; i++) {
        buffer_overflow_detected = 0;
        
        if (sigsetjmp(jump_buffer, 1) == 0) {
            /* Call the actual Ifx_Console_print function */
            Ifx_Console_print("%s", payloads[i]);
        }
        
        /* Assert no buffer overflow/crash occurred */
        ck_assert_msg(buffer_overflow_detected == 0,
                      "Buffer overflow detected with payload size %zu",
                      strlen(payloads[i]));
    }
    
    sigaction(SIGSEGV, &old_sa, NULL);
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_console_print_buffer_bounds);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}