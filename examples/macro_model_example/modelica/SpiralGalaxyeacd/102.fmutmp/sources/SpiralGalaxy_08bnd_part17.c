#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 13501
type: SIMPLE_ASSIGN
r_init[250] = r_min + 250.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13501};
  (data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (250.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13502
type: SIMPLE_ASSIGN
omega_c[250] = sqrt(G * Md / (r_init[250] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13502(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13502};
  modelica_real tmp2500;
  modelica_real tmp2501;
  modelica_real tmp2502;
  modelica_real tmp2503;
  modelica_real tmp2504;
  modelica_real tmp2505;
  modelica_real tmp2506;
  modelica_real tmp2507;
  modelica_real tmp2508;
  modelica_real tmp2509;
  tmp2500 = (data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */);
  tmp2501 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2502 = (tmp2500 * tmp2500) + (tmp2501 * tmp2501);
  tmp2503 = 1.5;
  if(tmp2502 < 0.0 && tmp2503 != 0.0)
  {
    tmp2505 = modf(tmp2503, &tmp2506);
    
    if(tmp2505 > 0.5)
    {
      tmp2505 -= 1.0;
      tmp2506 += 1.0;
    }
    else if(tmp2505 < -0.5)
    {
      tmp2505 += 1.0;
      tmp2506 -= 1.0;
    }
    
    if(fabs(tmp2505) < 1e-10)
      tmp2504 = pow(tmp2502, tmp2506);
    else
    {
      tmp2508 = modf(1.0/tmp2503, &tmp2507);
      if(tmp2508 > 0.5)
      {
        tmp2508 -= 1.0;
        tmp2507 += 1.0;
      }
      else if(tmp2508 < -0.5)
      {
        tmp2508 += 1.0;
        tmp2507 -= 1.0;
      }
      if(fabs(tmp2508) < 1e-10 && ((unsigned long)tmp2507 & 1))
      {
        tmp2504 = -pow(-tmp2502, tmp2505)*pow(tmp2502, tmp2506);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2502, tmp2503);
      }
    }
  }
  else
  {
    tmp2504 = pow(tmp2502, tmp2503);
  }
  if(isnan(tmp2504) || isinf(tmp2504))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2502, tmp2503);
  }tmp2509 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2504,"(r_init[250] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2509 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[250] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2509);
    }
  }
  (data->simulationInfo->realParameter[754] /* omega_c[250] PARAM */) = sqrt(tmp2509);
  TRACE_POP
}

/*
equation index: 13503
type: SIMPLE_ASSIGN
r_init[249] = r_min + 249.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13503};
  (data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (249.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13504
type: SIMPLE_ASSIGN
omega_c[249] = sqrt(G * Md / (r_init[249] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13504(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13504};
  modelica_real tmp2510;
  modelica_real tmp2511;
  modelica_real tmp2512;
  modelica_real tmp2513;
  modelica_real tmp2514;
  modelica_real tmp2515;
  modelica_real tmp2516;
  modelica_real tmp2517;
  modelica_real tmp2518;
  modelica_real tmp2519;
  tmp2510 = (data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */);
  tmp2511 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2512 = (tmp2510 * tmp2510) + (tmp2511 * tmp2511);
  tmp2513 = 1.5;
  if(tmp2512 < 0.0 && tmp2513 != 0.0)
  {
    tmp2515 = modf(tmp2513, &tmp2516);
    
    if(tmp2515 > 0.5)
    {
      tmp2515 -= 1.0;
      tmp2516 += 1.0;
    }
    else if(tmp2515 < -0.5)
    {
      tmp2515 += 1.0;
      tmp2516 -= 1.0;
    }
    
    if(fabs(tmp2515) < 1e-10)
      tmp2514 = pow(tmp2512, tmp2516);
    else
    {
      tmp2518 = modf(1.0/tmp2513, &tmp2517);
      if(tmp2518 > 0.5)
      {
        tmp2518 -= 1.0;
        tmp2517 += 1.0;
      }
      else if(tmp2518 < -0.5)
      {
        tmp2518 += 1.0;
        tmp2517 -= 1.0;
      }
      if(fabs(tmp2518) < 1e-10 && ((unsigned long)tmp2517 & 1))
      {
        tmp2514 = -pow(-tmp2512, tmp2515)*pow(tmp2512, tmp2516);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2512, tmp2513);
      }
    }
  }
  else
  {
    tmp2514 = pow(tmp2512, tmp2513);
  }
  if(isnan(tmp2514) || isinf(tmp2514))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2512, tmp2513);
  }tmp2519 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2514,"(r_init[249] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2519 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[249] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2519);
    }
  }
  (data->simulationInfo->realParameter[753] /* omega_c[249] PARAM */) = sqrt(tmp2519);
  TRACE_POP
}

/*
equation index: 13505
type: SIMPLE_ASSIGN
r_init[248] = r_min + 248.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13505};
  (data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (248.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13506
type: SIMPLE_ASSIGN
omega_c[248] = sqrt(G * Md / (r_init[248] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13506(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13506};
  modelica_real tmp2520;
  modelica_real tmp2521;
  modelica_real tmp2522;
  modelica_real tmp2523;
  modelica_real tmp2524;
  modelica_real tmp2525;
  modelica_real tmp2526;
  modelica_real tmp2527;
  modelica_real tmp2528;
  modelica_real tmp2529;
  tmp2520 = (data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */);
  tmp2521 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2522 = (tmp2520 * tmp2520) + (tmp2521 * tmp2521);
  tmp2523 = 1.5;
  if(tmp2522 < 0.0 && tmp2523 != 0.0)
  {
    tmp2525 = modf(tmp2523, &tmp2526);
    
    if(tmp2525 > 0.5)
    {
      tmp2525 -= 1.0;
      tmp2526 += 1.0;
    }
    else if(tmp2525 < -0.5)
    {
      tmp2525 += 1.0;
      tmp2526 -= 1.0;
    }
    
    if(fabs(tmp2525) < 1e-10)
      tmp2524 = pow(tmp2522, tmp2526);
    else
    {
      tmp2528 = modf(1.0/tmp2523, &tmp2527);
      if(tmp2528 > 0.5)
      {
        tmp2528 -= 1.0;
        tmp2527 += 1.0;
      }
      else if(tmp2528 < -0.5)
      {
        tmp2528 += 1.0;
        tmp2527 -= 1.0;
      }
      if(fabs(tmp2528) < 1e-10 && ((unsigned long)tmp2527 & 1))
      {
        tmp2524 = -pow(-tmp2522, tmp2525)*pow(tmp2522, tmp2526);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2522, tmp2523);
      }
    }
  }
  else
  {
    tmp2524 = pow(tmp2522, tmp2523);
  }
  if(isnan(tmp2524) || isinf(tmp2524))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2522, tmp2523);
  }tmp2529 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2524,"(r_init[248] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2529 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[248] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2529);
    }
  }
  (data->simulationInfo->realParameter[752] /* omega_c[248] PARAM */) = sqrt(tmp2529);
  TRACE_POP
}

/*
equation index: 13507
type: SIMPLE_ASSIGN
r_init[247] = r_min + 247.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13507};
  (data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (247.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13508
type: SIMPLE_ASSIGN
omega_c[247] = sqrt(G * Md / (r_init[247] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13508};
  modelica_real tmp2530;
  modelica_real tmp2531;
  modelica_real tmp2532;
  modelica_real tmp2533;
  modelica_real tmp2534;
  modelica_real tmp2535;
  modelica_real tmp2536;
  modelica_real tmp2537;
  modelica_real tmp2538;
  modelica_real tmp2539;
  tmp2530 = (data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */);
  tmp2531 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2532 = (tmp2530 * tmp2530) + (tmp2531 * tmp2531);
  tmp2533 = 1.5;
  if(tmp2532 < 0.0 && tmp2533 != 0.0)
  {
    tmp2535 = modf(tmp2533, &tmp2536);
    
    if(tmp2535 > 0.5)
    {
      tmp2535 -= 1.0;
      tmp2536 += 1.0;
    }
    else if(tmp2535 < -0.5)
    {
      tmp2535 += 1.0;
      tmp2536 -= 1.0;
    }
    
    if(fabs(tmp2535) < 1e-10)
      tmp2534 = pow(tmp2532, tmp2536);
    else
    {
      tmp2538 = modf(1.0/tmp2533, &tmp2537);
      if(tmp2538 > 0.5)
      {
        tmp2538 -= 1.0;
        tmp2537 += 1.0;
      }
      else if(tmp2538 < -0.5)
      {
        tmp2538 += 1.0;
        tmp2537 -= 1.0;
      }
      if(fabs(tmp2538) < 1e-10 && ((unsigned long)tmp2537 & 1))
      {
        tmp2534 = -pow(-tmp2532, tmp2535)*pow(tmp2532, tmp2536);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2532, tmp2533);
      }
    }
  }
  else
  {
    tmp2534 = pow(tmp2532, tmp2533);
  }
  if(isnan(tmp2534) || isinf(tmp2534))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2532, tmp2533);
  }tmp2539 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2534,"(r_init[247] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2539 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[247] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2539);
    }
  }
  (data->simulationInfo->realParameter[751] /* omega_c[247] PARAM */) = sqrt(tmp2539);
  TRACE_POP
}

/*
equation index: 13509
type: SIMPLE_ASSIGN
r_init[246] = r_min + 246.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13509};
  (data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (246.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13510
type: SIMPLE_ASSIGN
omega_c[246] = sqrt(G * Md / (r_init[246] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13510(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13510};
  modelica_real tmp2540;
  modelica_real tmp2541;
  modelica_real tmp2542;
  modelica_real tmp2543;
  modelica_real tmp2544;
  modelica_real tmp2545;
  modelica_real tmp2546;
  modelica_real tmp2547;
  modelica_real tmp2548;
  modelica_real tmp2549;
  tmp2540 = (data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */);
  tmp2541 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2542 = (tmp2540 * tmp2540) + (tmp2541 * tmp2541);
  tmp2543 = 1.5;
  if(tmp2542 < 0.0 && tmp2543 != 0.0)
  {
    tmp2545 = modf(tmp2543, &tmp2546);
    
    if(tmp2545 > 0.5)
    {
      tmp2545 -= 1.0;
      tmp2546 += 1.0;
    }
    else if(tmp2545 < -0.5)
    {
      tmp2545 += 1.0;
      tmp2546 -= 1.0;
    }
    
    if(fabs(tmp2545) < 1e-10)
      tmp2544 = pow(tmp2542, tmp2546);
    else
    {
      tmp2548 = modf(1.0/tmp2543, &tmp2547);
      if(tmp2548 > 0.5)
      {
        tmp2548 -= 1.0;
        tmp2547 += 1.0;
      }
      else if(tmp2548 < -0.5)
      {
        tmp2548 += 1.0;
        tmp2547 -= 1.0;
      }
      if(fabs(tmp2548) < 1e-10 && ((unsigned long)tmp2547 & 1))
      {
        tmp2544 = -pow(-tmp2542, tmp2545)*pow(tmp2542, tmp2546);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2542, tmp2543);
      }
    }
  }
  else
  {
    tmp2544 = pow(tmp2542, tmp2543);
  }
  if(isnan(tmp2544) || isinf(tmp2544))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2542, tmp2543);
  }tmp2549 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2544,"(r_init[246] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2549 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[246] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2549);
    }
  }
  (data->simulationInfo->realParameter[750] /* omega_c[246] PARAM */) = sqrt(tmp2549);
  TRACE_POP
}

/*
equation index: 13511
type: SIMPLE_ASSIGN
r_init[245] = r_min + 245.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13511};
  (data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (245.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13512
type: SIMPLE_ASSIGN
omega_c[245] = sqrt(G * Md / (r_init[245] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13512(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13512};
  modelica_real tmp2550;
  modelica_real tmp2551;
  modelica_real tmp2552;
  modelica_real tmp2553;
  modelica_real tmp2554;
  modelica_real tmp2555;
  modelica_real tmp2556;
  modelica_real tmp2557;
  modelica_real tmp2558;
  modelica_real tmp2559;
  tmp2550 = (data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */);
  tmp2551 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2552 = (tmp2550 * tmp2550) + (tmp2551 * tmp2551);
  tmp2553 = 1.5;
  if(tmp2552 < 0.0 && tmp2553 != 0.0)
  {
    tmp2555 = modf(tmp2553, &tmp2556);
    
    if(tmp2555 > 0.5)
    {
      tmp2555 -= 1.0;
      tmp2556 += 1.0;
    }
    else if(tmp2555 < -0.5)
    {
      tmp2555 += 1.0;
      tmp2556 -= 1.0;
    }
    
    if(fabs(tmp2555) < 1e-10)
      tmp2554 = pow(tmp2552, tmp2556);
    else
    {
      tmp2558 = modf(1.0/tmp2553, &tmp2557);
      if(tmp2558 > 0.5)
      {
        tmp2558 -= 1.0;
        tmp2557 += 1.0;
      }
      else if(tmp2558 < -0.5)
      {
        tmp2558 += 1.0;
        tmp2557 -= 1.0;
      }
      if(fabs(tmp2558) < 1e-10 && ((unsigned long)tmp2557 & 1))
      {
        tmp2554 = -pow(-tmp2552, tmp2555)*pow(tmp2552, tmp2556);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2552, tmp2553);
      }
    }
  }
  else
  {
    tmp2554 = pow(tmp2552, tmp2553);
  }
  if(isnan(tmp2554) || isinf(tmp2554))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2552, tmp2553);
  }tmp2559 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2554,"(r_init[245] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2559 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[245] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2559);
    }
  }
  (data->simulationInfo->realParameter[749] /* omega_c[245] PARAM */) = sqrt(tmp2559);
  TRACE_POP
}

/*
equation index: 13513
type: SIMPLE_ASSIGN
r_init[244] = r_min + 244.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13513};
  (data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (244.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13514
type: SIMPLE_ASSIGN
omega_c[244] = sqrt(G * Md / (r_init[244] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13514(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13514};
  modelica_real tmp2560;
  modelica_real tmp2561;
  modelica_real tmp2562;
  modelica_real tmp2563;
  modelica_real tmp2564;
  modelica_real tmp2565;
  modelica_real tmp2566;
  modelica_real tmp2567;
  modelica_real tmp2568;
  modelica_real tmp2569;
  tmp2560 = (data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */);
  tmp2561 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2562 = (tmp2560 * tmp2560) + (tmp2561 * tmp2561);
  tmp2563 = 1.5;
  if(tmp2562 < 0.0 && tmp2563 != 0.0)
  {
    tmp2565 = modf(tmp2563, &tmp2566);
    
    if(tmp2565 > 0.5)
    {
      tmp2565 -= 1.0;
      tmp2566 += 1.0;
    }
    else if(tmp2565 < -0.5)
    {
      tmp2565 += 1.0;
      tmp2566 -= 1.0;
    }
    
    if(fabs(tmp2565) < 1e-10)
      tmp2564 = pow(tmp2562, tmp2566);
    else
    {
      tmp2568 = modf(1.0/tmp2563, &tmp2567);
      if(tmp2568 > 0.5)
      {
        tmp2568 -= 1.0;
        tmp2567 += 1.0;
      }
      else if(tmp2568 < -0.5)
      {
        tmp2568 += 1.0;
        tmp2567 -= 1.0;
      }
      if(fabs(tmp2568) < 1e-10 && ((unsigned long)tmp2567 & 1))
      {
        tmp2564 = -pow(-tmp2562, tmp2565)*pow(tmp2562, tmp2566);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2562, tmp2563);
      }
    }
  }
  else
  {
    tmp2564 = pow(tmp2562, tmp2563);
  }
  if(isnan(tmp2564) || isinf(tmp2564))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2562, tmp2563);
  }tmp2569 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2564,"(r_init[244] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2569 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[244] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2569);
    }
  }
  (data->simulationInfo->realParameter[748] /* omega_c[244] PARAM */) = sqrt(tmp2569);
  TRACE_POP
}

/*
equation index: 13515
type: SIMPLE_ASSIGN
r_init[243] = r_min + 243.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13515};
  (data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (243.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13516
type: SIMPLE_ASSIGN
omega_c[243] = sqrt(G * Md / (r_init[243] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13516};
  modelica_real tmp2570;
  modelica_real tmp2571;
  modelica_real tmp2572;
  modelica_real tmp2573;
  modelica_real tmp2574;
  modelica_real tmp2575;
  modelica_real tmp2576;
  modelica_real tmp2577;
  modelica_real tmp2578;
  modelica_real tmp2579;
  tmp2570 = (data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */);
  tmp2571 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2572 = (tmp2570 * tmp2570) + (tmp2571 * tmp2571);
  tmp2573 = 1.5;
  if(tmp2572 < 0.0 && tmp2573 != 0.0)
  {
    tmp2575 = modf(tmp2573, &tmp2576);
    
    if(tmp2575 > 0.5)
    {
      tmp2575 -= 1.0;
      tmp2576 += 1.0;
    }
    else if(tmp2575 < -0.5)
    {
      tmp2575 += 1.0;
      tmp2576 -= 1.0;
    }
    
    if(fabs(tmp2575) < 1e-10)
      tmp2574 = pow(tmp2572, tmp2576);
    else
    {
      tmp2578 = modf(1.0/tmp2573, &tmp2577);
      if(tmp2578 > 0.5)
      {
        tmp2578 -= 1.0;
        tmp2577 += 1.0;
      }
      else if(tmp2578 < -0.5)
      {
        tmp2578 += 1.0;
        tmp2577 -= 1.0;
      }
      if(fabs(tmp2578) < 1e-10 && ((unsigned long)tmp2577 & 1))
      {
        tmp2574 = -pow(-tmp2572, tmp2575)*pow(tmp2572, tmp2576);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2572, tmp2573);
      }
    }
  }
  else
  {
    tmp2574 = pow(tmp2572, tmp2573);
  }
  if(isnan(tmp2574) || isinf(tmp2574))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2572, tmp2573);
  }tmp2579 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2574,"(r_init[243] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2579 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[243] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2579);
    }
  }
  (data->simulationInfo->realParameter[747] /* omega_c[243] PARAM */) = sqrt(tmp2579);
  TRACE_POP
}

/*
equation index: 13517
type: SIMPLE_ASSIGN
r_init[242] = r_min + 242.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13517};
  (data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (242.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13518
type: SIMPLE_ASSIGN
omega_c[242] = sqrt(G * Md / (r_init[242] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13518(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13518};
  modelica_real tmp2580;
  modelica_real tmp2581;
  modelica_real tmp2582;
  modelica_real tmp2583;
  modelica_real tmp2584;
  modelica_real tmp2585;
  modelica_real tmp2586;
  modelica_real tmp2587;
  modelica_real tmp2588;
  modelica_real tmp2589;
  tmp2580 = (data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */);
  tmp2581 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2582 = (tmp2580 * tmp2580) + (tmp2581 * tmp2581);
  tmp2583 = 1.5;
  if(tmp2582 < 0.0 && tmp2583 != 0.0)
  {
    tmp2585 = modf(tmp2583, &tmp2586);
    
    if(tmp2585 > 0.5)
    {
      tmp2585 -= 1.0;
      tmp2586 += 1.0;
    }
    else if(tmp2585 < -0.5)
    {
      tmp2585 += 1.0;
      tmp2586 -= 1.0;
    }
    
    if(fabs(tmp2585) < 1e-10)
      tmp2584 = pow(tmp2582, tmp2586);
    else
    {
      tmp2588 = modf(1.0/tmp2583, &tmp2587);
      if(tmp2588 > 0.5)
      {
        tmp2588 -= 1.0;
        tmp2587 += 1.0;
      }
      else if(tmp2588 < -0.5)
      {
        tmp2588 += 1.0;
        tmp2587 -= 1.0;
      }
      if(fabs(tmp2588) < 1e-10 && ((unsigned long)tmp2587 & 1))
      {
        tmp2584 = -pow(-tmp2582, tmp2585)*pow(tmp2582, tmp2586);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2582, tmp2583);
      }
    }
  }
  else
  {
    tmp2584 = pow(tmp2582, tmp2583);
  }
  if(isnan(tmp2584) || isinf(tmp2584))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2582, tmp2583);
  }tmp2589 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2584,"(r_init[242] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2589 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[242] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2589);
    }
  }
  (data->simulationInfo->realParameter[746] /* omega_c[242] PARAM */) = sqrt(tmp2589);
  TRACE_POP
}

/*
equation index: 13519
type: SIMPLE_ASSIGN
r_init[241] = r_min + 241.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13519};
  (data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (241.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13520
type: SIMPLE_ASSIGN
omega_c[241] = sqrt(G * Md / (r_init[241] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13520(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13520};
  modelica_real tmp2590;
  modelica_real tmp2591;
  modelica_real tmp2592;
  modelica_real tmp2593;
  modelica_real tmp2594;
  modelica_real tmp2595;
  modelica_real tmp2596;
  modelica_real tmp2597;
  modelica_real tmp2598;
  modelica_real tmp2599;
  tmp2590 = (data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */);
  tmp2591 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2592 = (tmp2590 * tmp2590) + (tmp2591 * tmp2591);
  tmp2593 = 1.5;
  if(tmp2592 < 0.0 && tmp2593 != 0.0)
  {
    tmp2595 = modf(tmp2593, &tmp2596);
    
    if(tmp2595 > 0.5)
    {
      tmp2595 -= 1.0;
      tmp2596 += 1.0;
    }
    else if(tmp2595 < -0.5)
    {
      tmp2595 += 1.0;
      tmp2596 -= 1.0;
    }
    
    if(fabs(tmp2595) < 1e-10)
      tmp2594 = pow(tmp2592, tmp2596);
    else
    {
      tmp2598 = modf(1.0/tmp2593, &tmp2597);
      if(tmp2598 > 0.5)
      {
        tmp2598 -= 1.0;
        tmp2597 += 1.0;
      }
      else if(tmp2598 < -0.5)
      {
        tmp2598 += 1.0;
        tmp2597 -= 1.0;
      }
      if(fabs(tmp2598) < 1e-10 && ((unsigned long)tmp2597 & 1))
      {
        tmp2594 = -pow(-tmp2592, tmp2595)*pow(tmp2592, tmp2596);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2592, tmp2593);
      }
    }
  }
  else
  {
    tmp2594 = pow(tmp2592, tmp2593);
  }
  if(isnan(tmp2594) || isinf(tmp2594))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2592, tmp2593);
  }tmp2599 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2594,"(r_init[241] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2599 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[241] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2599);
    }
  }
  (data->simulationInfo->realParameter[745] /* omega_c[241] PARAM */) = sqrt(tmp2599);
  TRACE_POP
}

/*
equation index: 13521
type: SIMPLE_ASSIGN
r_init[240] = r_min + 240.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13521};
  (data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (240.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13522
type: SIMPLE_ASSIGN
omega_c[240] = sqrt(G * Md / (r_init[240] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13522(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13522};
  modelica_real tmp2600;
  modelica_real tmp2601;
  modelica_real tmp2602;
  modelica_real tmp2603;
  modelica_real tmp2604;
  modelica_real tmp2605;
  modelica_real tmp2606;
  modelica_real tmp2607;
  modelica_real tmp2608;
  modelica_real tmp2609;
  tmp2600 = (data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */);
  tmp2601 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2602 = (tmp2600 * tmp2600) + (tmp2601 * tmp2601);
  tmp2603 = 1.5;
  if(tmp2602 < 0.0 && tmp2603 != 0.0)
  {
    tmp2605 = modf(tmp2603, &tmp2606);
    
    if(tmp2605 > 0.5)
    {
      tmp2605 -= 1.0;
      tmp2606 += 1.0;
    }
    else if(tmp2605 < -0.5)
    {
      tmp2605 += 1.0;
      tmp2606 -= 1.0;
    }
    
    if(fabs(tmp2605) < 1e-10)
      tmp2604 = pow(tmp2602, tmp2606);
    else
    {
      tmp2608 = modf(1.0/tmp2603, &tmp2607);
      if(tmp2608 > 0.5)
      {
        tmp2608 -= 1.0;
        tmp2607 += 1.0;
      }
      else if(tmp2608 < -0.5)
      {
        tmp2608 += 1.0;
        tmp2607 -= 1.0;
      }
      if(fabs(tmp2608) < 1e-10 && ((unsigned long)tmp2607 & 1))
      {
        tmp2604 = -pow(-tmp2602, tmp2605)*pow(tmp2602, tmp2606);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2602, tmp2603);
      }
    }
  }
  else
  {
    tmp2604 = pow(tmp2602, tmp2603);
  }
  if(isnan(tmp2604) || isinf(tmp2604))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2602, tmp2603);
  }tmp2609 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2604,"(r_init[240] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2609 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[240] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2609);
    }
  }
  (data->simulationInfo->realParameter[744] /* omega_c[240] PARAM */) = sqrt(tmp2609);
  TRACE_POP
}

/*
equation index: 13523
type: SIMPLE_ASSIGN
r_init[239] = r_min + 239.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13523};
  (data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (239.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13524
type: SIMPLE_ASSIGN
omega_c[239] = sqrt(G * Md / (r_init[239] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13524};
  modelica_real tmp2610;
  modelica_real tmp2611;
  modelica_real tmp2612;
  modelica_real tmp2613;
  modelica_real tmp2614;
  modelica_real tmp2615;
  modelica_real tmp2616;
  modelica_real tmp2617;
  modelica_real tmp2618;
  modelica_real tmp2619;
  tmp2610 = (data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */);
  tmp2611 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2612 = (tmp2610 * tmp2610) + (tmp2611 * tmp2611);
  tmp2613 = 1.5;
  if(tmp2612 < 0.0 && tmp2613 != 0.0)
  {
    tmp2615 = modf(tmp2613, &tmp2616);
    
    if(tmp2615 > 0.5)
    {
      tmp2615 -= 1.0;
      tmp2616 += 1.0;
    }
    else if(tmp2615 < -0.5)
    {
      tmp2615 += 1.0;
      tmp2616 -= 1.0;
    }
    
    if(fabs(tmp2615) < 1e-10)
      tmp2614 = pow(tmp2612, tmp2616);
    else
    {
      tmp2618 = modf(1.0/tmp2613, &tmp2617);
      if(tmp2618 > 0.5)
      {
        tmp2618 -= 1.0;
        tmp2617 += 1.0;
      }
      else if(tmp2618 < -0.5)
      {
        tmp2618 += 1.0;
        tmp2617 -= 1.0;
      }
      if(fabs(tmp2618) < 1e-10 && ((unsigned long)tmp2617 & 1))
      {
        tmp2614 = -pow(-tmp2612, tmp2615)*pow(tmp2612, tmp2616);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2612, tmp2613);
      }
    }
  }
  else
  {
    tmp2614 = pow(tmp2612, tmp2613);
  }
  if(isnan(tmp2614) || isinf(tmp2614))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2612, tmp2613);
  }tmp2619 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2614,"(r_init[239] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2619 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[239] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2619);
    }
  }
  (data->simulationInfo->realParameter[743] /* omega_c[239] PARAM */) = sqrt(tmp2619);
  TRACE_POP
}

/*
equation index: 13525
type: SIMPLE_ASSIGN
r_init[238] = r_min + 238.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13525};
  (data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (238.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13526
type: SIMPLE_ASSIGN
omega_c[238] = sqrt(G * Md / (r_init[238] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13526(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13526};
  modelica_real tmp2620;
  modelica_real tmp2621;
  modelica_real tmp2622;
  modelica_real tmp2623;
  modelica_real tmp2624;
  modelica_real tmp2625;
  modelica_real tmp2626;
  modelica_real tmp2627;
  modelica_real tmp2628;
  modelica_real tmp2629;
  tmp2620 = (data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */);
  tmp2621 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2622 = (tmp2620 * tmp2620) + (tmp2621 * tmp2621);
  tmp2623 = 1.5;
  if(tmp2622 < 0.0 && tmp2623 != 0.0)
  {
    tmp2625 = modf(tmp2623, &tmp2626);
    
    if(tmp2625 > 0.5)
    {
      tmp2625 -= 1.0;
      tmp2626 += 1.0;
    }
    else if(tmp2625 < -0.5)
    {
      tmp2625 += 1.0;
      tmp2626 -= 1.0;
    }
    
    if(fabs(tmp2625) < 1e-10)
      tmp2624 = pow(tmp2622, tmp2626);
    else
    {
      tmp2628 = modf(1.0/tmp2623, &tmp2627);
      if(tmp2628 > 0.5)
      {
        tmp2628 -= 1.0;
        tmp2627 += 1.0;
      }
      else if(tmp2628 < -0.5)
      {
        tmp2628 += 1.0;
        tmp2627 -= 1.0;
      }
      if(fabs(tmp2628) < 1e-10 && ((unsigned long)tmp2627 & 1))
      {
        tmp2624 = -pow(-tmp2622, tmp2625)*pow(tmp2622, tmp2626);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2622, tmp2623);
      }
    }
  }
  else
  {
    tmp2624 = pow(tmp2622, tmp2623);
  }
  if(isnan(tmp2624) || isinf(tmp2624))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2622, tmp2623);
  }tmp2629 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2624,"(r_init[238] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2629 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[238] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2629);
    }
  }
  (data->simulationInfo->realParameter[742] /* omega_c[238] PARAM */) = sqrt(tmp2629);
  TRACE_POP
}

/*
equation index: 13527
type: SIMPLE_ASSIGN
r_init[237] = r_min + 237.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13527};
  (data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (237.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13528
type: SIMPLE_ASSIGN
omega_c[237] = sqrt(G * Md / (r_init[237] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13528(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13528};
  modelica_real tmp2630;
  modelica_real tmp2631;
  modelica_real tmp2632;
  modelica_real tmp2633;
  modelica_real tmp2634;
  modelica_real tmp2635;
  modelica_real tmp2636;
  modelica_real tmp2637;
  modelica_real tmp2638;
  modelica_real tmp2639;
  tmp2630 = (data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */);
  tmp2631 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2632 = (tmp2630 * tmp2630) + (tmp2631 * tmp2631);
  tmp2633 = 1.5;
  if(tmp2632 < 0.0 && tmp2633 != 0.0)
  {
    tmp2635 = modf(tmp2633, &tmp2636);
    
    if(tmp2635 > 0.5)
    {
      tmp2635 -= 1.0;
      tmp2636 += 1.0;
    }
    else if(tmp2635 < -0.5)
    {
      tmp2635 += 1.0;
      tmp2636 -= 1.0;
    }
    
    if(fabs(tmp2635) < 1e-10)
      tmp2634 = pow(tmp2632, tmp2636);
    else
    {
      tmp2638 = modf(1.0/tmp2633, &tmp2637);
      if(tmp2638 > 0.5)
      {
        tmp2638 -= 1.0;
        tmp2637 += 1.0;
      }
      else if(tmp2638 < -0.5)
      {
        tmp2638 += 1.0;
        tmp2637 -= 1.0;
      }
      if(fabs(tmp2638) < 1e-10 && ((unsigned long)tmp2637 & 1))
      {
        tmp2634 = -pow(-tmp2632, tmp2635)*pow(tmp2632, tmp2636);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2632, tmp2633);
      }
    }
  }
  else
  {
    tmp2634 = pow(tmp2632, tmp2633);
  }
  if(isnan(tmp2634) || isinf(tmp2634))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2632, tmp2633);
  }tmp2639 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2634,"(r_init[237] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2639 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[237] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2639);
    }
  }
  (data->simulationInfo->realParameter[741] /* omega_c[237] PARAM */) = sqrt(tmp2639);
  TRACE_POP
}

/*
equation index: 13529
type: SIMPLE_ASSIGN
r_init[236] = r_min + 236.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13529};
  (data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (236.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13530
type: SIMPLE_ASSIGN
omega_c[236] = sqrt(G * Md / (r_init[236] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13530(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13530};
  modelica_real tmp2640;
  modelica_real tmp2641;
  modelica_real tmp2642;
  modelica_real tmp2643;
  modelica_real tmp2644;
  modelica_real tmp2645;
  modelica_real tmp2646;
  modelica_real tmp2647;
  modelica_real tmp2648;
  modelica_real tmp2649;
  tmp2640 = (data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */);
  tmp2641 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2642 = (tmp2640 * tmp2640) + (tmp2641 * tmp2641);
  tmp2643 = 1.5;
  if(tmp2642 < 0.0 && tmp2643 != 0.0)
  {
    tmp2645 = modf(tmp2643, &tmp2646);
    
    if(tmp2645 > 0.5)
    {
      tmp2645 -= 1.0;
      tmp2646 += 1.0;
    }
    else if(tmp2645 < -0.5)
    {
      tmp2645 += 1.0;
      tmp2646 -= 1.0;
    }
    
    if(fabs(tmp2645) < 1e-10)
      tmp2644 = pow(tmp2642, tmp2646);
    else
    {
      tmp2648 = modf(1.0/tmp2643, &tmp2647);
      if(tmp2648 > 0.5)
      {
        tmp2648 -= 1.0;
        tmp2647 += 1.0;
      }
      else if(tmp2648 < -0.5)
      {
        tmp2648 += 1.0;
        tmp2647 -= 1.0;
      }
      if(fabs(tmp2648) < 1e-10 && ((unsigned long)tmp2647 & 1))
      {
        tmp2644 = -pow(-tmp2642, tmp2645)*pow(tmp2642, tmp2646);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2642, tmp2643);
      }
    }
  }
  else
  {
    tmp2644 = pow(tmp2642, tmp2643);
  }
  if(isnan(tmp2644) || isinf(tmp2644))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2642, tmp2643);
  }tmp2649 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2644,"(r_init[236] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2649 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[236] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2649);
    }
  }
  (data->simulationInfo->realParameter[740] /* omega_c[236] PARAM */) = sqrt(tmp2649);
  TRACE_POP
}

/*
equation index: 13531
type: SIMPLE_ASSIGN
r_init[235] = r_min + 235.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13531};
  (data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (235.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13532
type: SIMPLE_ASSIGN
omega_c[235] = sqrt(G * Md / (r_init[235] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13532};
  modelica_real tmp2650;
  modelica_real tmp2651;
  modelica_real tmp2652;
  modelica_real tmp2653;
  modelica_real tmp2654;
  modelica_real tmp2655;
  modelica_real tmp2656;
  modelica_real tmp2657;
  modelica_real tmp2658;
  modelica_real tmp2659;
  tmp2650 = (data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */);
  tmp2651 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2652 = (tmp2650 * tmp2650) + (tmp2651 * tmp2651);
  tmp2653 = 1.5;
  if(tmp2652 < 0.0 && tmp2653 != 0.0)
  {
    tmp2655 = modf(tmp2653, &tmp2656);
    
    if(tmp2655 > 0.5)
    {
      tmp2655 -= 1.0;
      tmp2656 += 1.0;
    }
    else if(tmp2655 < -0.5)
    {
      tmp2655 += 1.0;
      tmp2656 -= 1.0;
    }
    
    if(fabs(tmp2655) < 1e-10)
      tmp2654 = pow(tmp2652, tmp2656);
    else
    {
      tmp2658 = modf(1.0/tmp2653, &tmp2657);
      if(tmp2658 > 0.5)
      {
        tmp2658 -= 1.0;
        tmp2657 += 1.0;
      }
      else if(tmp2658 < -0.5)
      {
        tmp2658 += 1.0;
        tmp2657 -= 1.0;
      }
      if(fabs(tmp2658) < 1e-10 && ((unsigned long)tmp2657 & 1))
      {
        tmp2654 = -pow(-tmp2652, tmp2655)*pow(tmp2652, tmp2656);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2652, tmp2653);
      }
    }
  }
  else
  {
    tmp2654 = pow(tmp2652, tmp2653);
  }
  if(isnan(tmp2654) || isinf(tmp2654))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2652, tmp2653);
  }tmp2659 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2654,"(r_init[235] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2659 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[235] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2659);
    }
  }
  (data->simulationInfo->realParameter[739] /* omega_c[235] PARAM */) = sqrt(tmp2659);
  TRACE_POP
}

/*
equation index: 13533
type: SIMPLE_ASSIGN
r_init[234] = r_min + 234.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13533};
  (data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (234.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13534
type: SIMPLE_ASSIGN
omega_c[234] = sqrt(G * Md / (r_init[234] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13534(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13534};
  modelica_real tmp2660;
  modelica_real tmp2661;
  modelica_real tmp2662;
  modelica_real tmp2663;
  modelica_real tmp2664;
  modelica_real tmp2665;
  modelica_real tmp2666;
  modelica_real tmp2667;
  modelica_real tmp2668;
  modelica_real tmp2669;
  tmp2660 = (data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */);
  tmp2661 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2662 = (tmp2660 * tmp2660) + (tmp2661 * tmp2661);
  tmp2663 = 1.5;
  if(tmp2662 < 0.0 && tmp2663 != 0.0)
  {
    tmp2665 = modf(tmp2663, &tmp2666);
    
    if(tmp2665 > 0.5)
    {
      tmp2665 -= 1.0;
      tmp2666 += 1.0;
    }
    else if(tmp2665 < -0.5)
    {
      tmp2665 += 1.0;
      tmp2666 -= 1.0;
    }
    
    if(fabs(tmp2665) < 1e-10)
      tmp2664 = pow(tmp2662, tmp2666);
    else
    {
      tmp2668 = modf(1.0/tmp2663, &tmp2667);
      if(tmp2668 > 0.5)
      {
        tmp2668 -= 1.0;
        tmp2667 += 1.0;
      }
      else if(tmp2668 < -0.5)
      {
        tmp2668 += 1.0;
        tmp2667 -= 1.0;
      }
      if(fabs(tmp2668) < 1e-10 && ((unsigned long)tmp2667 & 1))
      {
        tmp2664 = -pow(-tmp2662, tmp2665)*pow(tmp2662, tmp2666);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2662, tmp2663);
      }
    }
  }
  else
  {
    tmp2664 = pow(tmp2662, tmp2663);
  }
  if(isnan(tmp2664) || isinf(tmp2664))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2662, tmp2663);
  }tmp2669 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2664,"(r_init[234] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2669 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[234] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2669);
    }
  }
  (data->simulationInfo->realParameter[738] /* omega_c[234] PARAM */) = sqrt(tmp2669);
  TRACE_POP
}

/*
equation index: 13535
type: SIMPLE_ASSIGN
r_init[233] = r_min + 233.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13535};
  (data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (233.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13536
type: SIMPLE_ASSIGN
omega_c[233] = sqrt(G * Md / (r_init[233] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13536(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13536};
  modelica_real tmp2670;
  modelica_real tmp2671;
  modelica_real tmp2672;
  modelica_real tmp2673;
  modelica_real tmp2674;
  modelica_real tmp2675;
  modelica_real tmp2676;
  modelica_real tmp2677;
  modelica_real tmp2678;
  modelica_real tmp2679;
  tmp2670 = (data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */);
  tmp2671 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2672 = (tmp2670 * tmp2670) + (tmp2671 * tmp2671);
  tmp2673 = 1.5;
  if(tmp2672 < 0.0 && tmp2673 != 0.0)
  {
    tmp2675 = modf(tmp2673, &tmp2676);
    
    if(tmp2675 > 0.5)
    {
      tmp2675 -= 1.0;
      tmp2676 += 1.0;
    }
    else if(tmp2675 < -0.5)
    {
      tmp2675 += 1.0;
      tmp2676 -= 1.0;
    }
    
    if(fabs(tmp2675) < 1e-10)
      tmp2674 = pow(tmp2672, tmp2676);
    else
    {
      tmp2678 = modf(1.0/tmp2673, &tmp2677);
      if(tmp2678 > 0.5)
      {
        tmp2678 -= 1.0;
        tmp2677 += 1.0;
      }
      else if(tmp2678 < -0.5)
      {
        tmp2678 += 1.0;
        tmp2677 -= 1.0;
      }
      if(fabs(tmp2678) < 1e-10 && ((unsigned long)tmp2677 & 1))
      {
        tmp2674 = -pow(-tmp2672, tmp2675)*pow(tmp2672, tmp2676);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2672, tmp2673);
      }
    }
  }
  else
  {
    tmp2674 = pow(tmp2672, tmp2673);
  }
  if(isnan(tmp2674) || isinf(tmp2674))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2672, tmp2673);
  }tmp2679 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2674,"(r_init[233] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2679 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[233] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2679);
    }
  }
  (data->simulationInfo->realParameter[737] /* omega_c[233] PARAM */) = sqrt(tmp2679);
  TRACE_POP
}

/*
equation index: 13537
type: SIMPLE_ASSIGN
r_init[232] = r_min + 232.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13537};
  (data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (232.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13538
type: SIMPLE_ASSIGN
omega_c[232] = sqrt(G * Md / (r_init[232] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13538(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13538};
  modelica_real tmp2680;
  modelica_real tmp2681;
  modelica_real tmp2682;
  modelica_real tmp2683;
  modelica_real tmp2684;
  modelica_real tmp2685;
  modelica_real tmp2686;
  modelica_real tmp2687;
  modelica_real tmp2688;
  modelica_real tmp2689;
  tmp2680 = (data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */);
  tmp2681 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2682 = (tmp2680 * tmp2680) + (tmp2681 * tmp2681);
  tmp2683 = 1.5;
  if(tmp2682 < 0.0 && tmp2683 != 0.0)
  {
    tmp2685 = modf(tmp2683, &tmp2686);
    
    if(tmp2685 > 0.5)
    {
      tmp2685 -= 1.0;
      tmp2686 += 1.0;
    }
    else if(tmp2685 < -0.5)
    {
      tmp2685 += 1.0;
      tmp2686 -= 1.0;
    }
    
    if(fabs(tmp2685) < 1e-10)
      tmp2684 = pow(tmp2682, tmp2686);
    else
    {
      tmp2688 = modf(1.0/tmp2683, &tmp2687);
      if(tmp2688 > 0.5)
      {
        tmp2688 -= 1.0;
        tmp2687 += 1.0;
      }
      else if(tmp2688 < -0.5)
      {
        tmp2688 += 1.0;
        tmp2687 -= 1.0;
      }
      if(fabs(tmp2688) < 1e-10 && ((unsigned long)tmp2687 & 1))
      {
        tmp2684 = -pow(-tmp2682, tmp2685)*pow(tmp2682, tmp2686);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2682, tmp2683);
      }
    }
  }
  else
  {
    tmp2684 = pow(tmp2682, tmp2683);
  }
  if(isnan(tmp2684) || isinf(tmp2684))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2682, tmp2683);
  }tmp2689 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2684,"(r_init[232] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2689 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[232] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2689);
    }
  }
  (data->simulationInfo->realParameter[736] /* omega_c[232] PARAM */) = sqrt(tmp2689);
  TRACE_POP
}

/*
equation index: 13539
type: SIMPLE_ASSIGN
r_init[231] = r_min + 231.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13539};
  (data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (231.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13540
type: SIMPLE_ASSIGN
omega_c[231] = sqrt(G * Md / (r_init[231] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13540};
  modelica_real tmp2690;
  modelica_real tmp2691;
  modelica_real tmp2692;
  modelica_real tmp2693;
  modelica_real tmp2694;
  modelica_real tmp2695;
  modelica_real tmp2696;
  modelica_real tmp2697;
  modelica_real tmp2698;
  modelica_real tmp2699;
  tmp2690 = (data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */);
  tmp2691 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2692 = (tmp2690 * tmp2690) + (tmp2691 * tmp2691);
  tmp2693 = 1.5;
  if(tmp2692 < 0.0 && tmp2693 != 0.0)
  {
    tmp2695 = modf(tmp2693, &tmp2696);
    
    if(tmp2695 > 0.5)
    {
      tmp2695 -= 1.0;
      tmp2696 += 1.0;
    }
    else if(tmp2695 < -0.5)
    {
      tmp2695 += 1.0;
      tmp2696 -= 1.0;
    }
    
    if(fabs(tmp2695) < 1e-10)
      tmp2694 = pow(tmp2692, tmp2696);
    else
    {
      tmp2698 = modf(1.0/tmp2693, &tmp2697);
      if(tmp2698 > 0.5)
      {
        tmp2698 -= 1.0;
        tmp2697 += 1.0;
      }
      else if(tmp2698 < -0.5)
      {
        tmp2698 += 1.0;
        tmp2697 -= 1.0;
      }
      if(fabs(tmp2698) < 1e-10 && ((unsigned long)tmp2697 & 1))
      {
        tmp2694 = -pow(-tmp2692, tmp2695)*pow(tmp2692, tmp2696);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2692, tmp2693);
      }
    }
  }
  else
  {
    tmp2694 = pow(tmp2692, tmp2693);
  }
  if(isnan(tmp2694) || isinf(tmp2694))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2692, tmp2693);
  }tmp2699 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2694,"(r_init[231] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2699 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[231] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2699);
    }
  }
  (data->simulationInfo->realParameter[735] /* omega_c[231] PARAM */) = sqrt(tmp2699);
  TRACE_POP
}

/*
equation index: 13541
type: SIMPLE_ASSIGN
r_init[230] = r_min + 230.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13541};
  (data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (230.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13542
type: SIMPLE_ASSIGN
omega_c[230] = sqrt(G * Md / (r_init[230] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13542(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13542};
  modelica_real tmp2700;
  modelica_real tmp2701;
  modelica_real tmp2702;
  modelica_real tmp2703;
  modelica_real tmp2704;
  modelica_real tmp2705;
  modelica_real tmp2706;
  modelica_real tmp2707;
  modelica_real tmp2708;
  modelica_real tmp2709;
  tmp2700 = (data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */);
  tmp2701 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2702 = (tmp2700 * tmp2700) + (tmp2701 * tmp2701);
  tmp2703 = 1.5;
  if(tmp2702 < 0.0 && tmp2703 != 0.0)
  {
    tmp2705 = modf(tmp2703, &tmp2706);
    
    if(tmp2705 > 0.5)
    {
      tmp2705 -= 1.0;
      tmp2706 += 1.0;
    }
    else if(tmp2705 < -0.5)
    {
      tmp2705 += 1.0;
      tmp2706 -= 1.0;
    }
    
    if(fabs(tmp2705) < 1e-10)
      tmp2704 = pow(tmp2702, tmp2706);
    else
    {
      tmp2708 = modf(1.0/tmp2703, &tmp2707);
      if(tmp2708 > 0.5)
      {
        tmp2708 -= 1.0;
        tmp2707 += 1.0;
      }
      else if(tmp2708 < -0.5)
      {
        tmp2708 += 1.0;
        tmp2707 -= 1.0;
      }
      if(fabs(tmp2708) < 1e-10 && ((unsigned long)tmp2707 & 1))
      {
        tmp2704 = -pow(-tmp2702, tmp2705)*pow(tmp2702, tmp2706);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2702, tmp2703);
      }
    }
  }
  else
  {
    tmp2704 = pow(tmp2702, tmp2703);
  }
  if(isnan(tmp2704) || isinf(tmp2704))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2702, tmp2703);
  }tmp2709 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2704,"(r_init[230] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2709 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[230] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2709);
    }
  }
  (data->simulationInfo->realParameter[734] /* omega_c[230] PARAM */) = sqrt(tmp2709);
  TRACE_POP
}

/*
equation index: 13543
type: SIMPLE_ASSIGN
r_init[229] = r_min + 229.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13543};
  (data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (229.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13544
type: SIMPLE_ASSIGN
omega_c[229] = sqrt(G * Md / (r_init[229] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13544(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13544};
  modelica_real tmp2710;
  modelica_real tmp2711;
  modelica_real tmp2712;
  modelica_real tmp2713;
  modelica_real tmp2714;
  modelica_real tmp2715;
  modelica_real tmp2716;
  modelica_real tmp2717;
  modelica_real tmp2718;
  modelica_real tmp2719;
  tmp2710 = (data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */);
  tmp2711 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2712 = (tmp2710 * tmp2710) + (tmp2711 * tmp2711);
  tmp2713 = 1.5;
  if(tmp2712 < 0.0 && tmp2713 != 0.0)
  {
    tmp2715 = modf(tmp2713, &tmp2716);
    
    if(tmp2715 > 0.5)
    {
      tmp2715 -= 1.0;
      tmp2716 += 1.0;
    }
    else if(tmp2715 < -0.5)
    {
      tmp2715 += 1.0;
      tmp2716 -= 1.0;
    }
    
    if(fabs(tmp2715) < 1e-10)
      tmp2714 = pow(tmp2712, tmp2716);
    else
    {
      tmp2718 = modf(1.0/tmp2713, &tmp2717);
      if(tmp2718 > 0.5)
      {
        tmp2718 -= 1.0;
        tmp2717 += 1.0;
      }
      else if(tmp2718 < -0.5)
      {
        tmp2718 += 1.0;
        tmp2717 -= 1.0;
      }
      if(fabs(tmp2718) < 1e-10 && ((unsigned long)tmp2717 & 1))
      {
        tmp2714 = -pow(-tmp2712, tmp2715)*pow(tmp2712, tmp2716);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2712, tmp2713);
      }
    }
  }
  else
  {
    tmp2714 = pow(tmp2712, tmp2713);
  }
  if(isnan(tmp2714) || isinf(tmp2714))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2712, tmp2713);
  }tmp2719 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2714,"(r_init[229] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2719 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[229] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2719);
    }
  }
  (data->simulationInfo->realParameter[733] /* omega_c[229] PARAM */) = sqrt(tmp2719);
  TRACE_POP
}

/*
equation index: 13545
type: SIMPLE_ASSIGN
r_init[228] = r_min + 228.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13545};
  (data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (228.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13546
type: SIMPLE_ASSIGN
omega_c[228] = sqrt(G * Md / (r_init[228] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13546(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13546};
  modelica_real tmp2720;
  modelica_real tmp2721;
  modelica_real tmp2722;
  modelica_real tmp2723;
  modelica_real tmp2724;
  modelica_real tmp2725;
  modelica_real tmp2726;
  modelica_real tmp2727;
  modelica_real tmp2728;
  modelica_real tmp2729;
  tmp2720 = (data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */);
  tmp2721 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2722 = (tmp2720 * tmp2720) + (tmp2721 * tmp2721);
  tmp2723 = 1.5;
  if(tmp2722 < 0.0 && tmp2723 != 0.0)
  {
    tmp2725 = modf(tmp2723, &tmp2726);
    
    if(tmp2725 > 0.5)
    {
      tmp2725 -= 1.0;
      tmp2726 += 1.0;
    }
    else if(tmp2725 < -0.5)
    {
      tmp2725 += 1.0;
      tmp2726 -= 1.0;
    }
    
    if(fabs(tmp2725) < 1e-10)
      tmp2724 = pow(tmp2722, tmp2726);
    else
    {
      tmp2728 = modf(1.0/tmp2723, &tmp2727);
      if(tmp2728 > 0.5)
      {
        tmp2728 -= 1.0;
        tmp2727 += 1.0;
      }
      else if(tmp2728 < -0.5)
      {
        tmp2728 += 1.0;
        tmp2727 -= 1.0;
      }
      if(fabs(tmp2728) < 1e-10 && ((unsigned long)tmp2727 & 1))
      {
        tmp2724 = -pow(-tmp2722, tmp2725)*pow(tmp2722, tmp2726);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2722, tmp2723);
      }
    }
  }
  else
  {
    tmp2724 = pow(tmp2722, tmp2723);
  }
  if(isnan(tmp2724) || isinf(tmp2724))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2722, tmp2723);
  }tmp2729 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2724,"(r_init[228] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2729 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[228] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2729);
    }
  }
  (data->simulationInfo->realParameter[732] /* omega_c[228] PARAM */) = sqrt(tmp2729);
  TRACE_POP
}

/*
equation index: 13547
type: SIMPLE_ASSIGN
r_init[227] = r_min + 227.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13547};
  (data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (227.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13548
type: SIMPLE_ASSIGN
omega_c[227] = sqrt(G * Md / (r_init[227] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13548};
  modelica_real tmp2730;
  modelica_real tmp2731;
  modelica_real tmp2732;
  modelica_real tmp2733;
  modelica_real tmp2734;
  modelica_real tmp2735;
  modelica_real tmp2736;
  modelica_real tmp2737;
  modelica_real tmp2738;
  modelica_real tmp2739;
  tmp2730 = (data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */);
  tmp2731 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2732 = (tmp2730 * tmp2730) + (tmp2731 * tmp2731);
  tmp2733 = 1.5;
  if(tmp2732 < 0.0 && tmp2733 != 0.0)
  {
    tmp2735 = modf(tmp2733, &tmp2736);
    
    if(tmp2735 > 0.5)
    {
      tmp2735 -= 1.0;
      tmp2736 += 1.0;
    }
    else if(tmp2735 < -0.5)
    {
      tmp2735 += 1.0;
      tmp2736 -= 1.0;
    }
    
    if(fabs(tmp2735) < 1e-10)
      tmp2734 = pow(tmp2732, tmp2736);
    else
    {
      tmp2738 = modf(1.0/tmp2733, &tmp2737);
      if(tmp2738 > 0.5)
      {
        tmp2738 -= 1.0;
        tmp2737 += 1.0;
      }
      else if(tmp2738 < -0.5)
      {
        tmp2738 += 1.0;
        tmp2737 -= 1.0;
      }
      if(fabs(tmp2738) < 1e-10 && ((unsigned long)tmp2737 & 1))
      {
        tmp2734 = -pow(-tmp2732, tmp2735)*pow(tmp2732, tmp2736);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2732, tmp2733);
      }
    }
  }
  else
  {
    tmp2734 = pow(tmp2732, tmp2733);
  }
  if(isnan(tmp2734) || isinf(tmp2734))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2732, tmp2733);
  }tmp2739 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2734,"(r_init[227] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2739 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[227] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2739);
    }
  }
  (data->simulationInfo->realParameter[731] /* omega_c[227] PARAM */) = sqrt(tmp2739);
  TRACE_POP
}

/*
equation index: 13549
type: SIMPLE_ASSIGN
r_init[226] = r_min + 226.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13549};
  (data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (226.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13550
type: SIMPLE_ASSIGN
omega_c[226] = sqrt(G * Md / (r_init[226] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13550(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13550};
  modelica_real tmp2740;
  modelica_real tmp2741;
  modelica_real tmp2742;
  modelica_real tmp2743;
  modelica_real tmp2744;
  modelica_real tmp2745;
  modelica_real tmp2746;
  modelica_real tmp2747;
  modelica_real tmp2748;
  modelica_real tmp2749;
  tmp2740 = (data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */);
  tmp2741 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2742 = (tmp2740 * tmp2740) + (tmp2741 * tmp2741);
  tmp2743 = 1.5;
  if(tmp2742 < 0.0 && tmp2743 != 0.0)
  {
    tmp2745 = modf(tmp2743, &tmp2746);
    
    if(tmp2745 > 0.5)
    {
      tmp2745 -= 1.0;
      tmp2746 += 1.0;
    }
    else if(tmp2745 < -0.5)
    {
      tmp2745 += 1.0;
      tmp2746 -= 1.0;
    }
    
    if(fabs(tmp2745) < 1e-10)
      tmp2744 = pow(tmp2742, tmp2746);
    else
    {
      tmp2748 = modf(1.0/tmp2743, &tmp2747);
      if(tmp2748 > 0.5)
      {
        tmp2748 -= 1.0;
        tmp2747 += 1.0;
      }
      else if(tmp2748 < -0.5)
      {
        tmp2748 += 1.0;
        tmp2747 -= 1.0;
      }
      if(fabs(tmp2748) < 1e-10 && ((unsigned long)tmp2747 & 1))
      {
        tmp2744 = -pow(-tmp2742, tmp2745)*pow(tmp2742, tmp2746);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2742, tmp2743);
      }
    }
  }
  else
  {
    tmp2744 = pow(tmp2742, tmp2743);
  }
  if(isnan(tmp2744) || isinf(tmp2744))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2742, tmp2743);
  }tmp2749 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2744,"(r_init[226] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2749 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[226] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2749);
    }
  }
  (data->simulationInfo->realParameter[730] /* omega_c[226] PARAM */) = sqrt(tmp2749);
  TRACE_POP
}

/*
equation index: 13551
type: SIMPLE_ASSIGN
r_init[225] = r_min + 225.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13551};
  (data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (225.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13552
type: SIMPLE_ASSIGN
omega_c[225] = sqrt(G * Md / (r_init[225] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13552(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13552};
  modelica_real tmp2750;
  modelica_real tmp2751;
  modelica_real tmp2752;
  modelica_real tmp2753;
  modelica_real tmp2754;
  modelica_real tmp2755;
  modelica_real tmp2756;
  modelica_real tmp2757;
  modelica_real tmp2758;
  modelica_real tmp2759;
  tmp2750 = (data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */);
  tmp2751 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2752 = (tmp2750 * tmp2750) + (tmp2751 * tmp2751);
  tmp2753 = 1.5;
  if(tmp2752 < 0.0 && tmp2753 != 0.0)
  {
    tmp2755 = modf(tmp2753, &tmp2756);
    
    if(tmp2755 > 0.5)
    {
      tmp2755 -= 1.0;
      tmp2756 += 1.0;
    }
    else if(tmp2755 < -0.5)
    {
      tmp2755 += 1.0;
      tmp2756 -= 1.0;
    }
    
    if(fabs(tmp2755) < 1e-10)
      tmp2754 = pow(tmp2752, tmp2756);
    else
    {
      tmp2758 = modf(1.0/tmp2753, &tmp2757);
      if(tmp2758 > 0.5)
      {
        tmp2758 -= 1.0;
        tmp2757 += 1.0;
      }
      else if(tmp2758 < -0.5)
      {
        tmp2758 += 1.0;
        tmp2757 -= 1.0;
      }
      if(fabs(tmp2758) < 1e-10 && ((unsigned long)tmp2757 & 1))
      {
        tmp2754 = -pow(-tmp2752, tmp2755)*pow(tmp2752, tmp2756);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2752, tmp2753);
      }
    }
  }
  else
  {
    tmp2754 = pow(tmp2752, tmp2753);
  }
  if(isnan(tmp2754) || isinf(tmp2754))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2752, tmp2753);
  }tmp2759 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2754,"(r_init[225] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2759 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[225] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2759);
    }
  }
  (data->simulationInfo->realParameter[729] /* omega_c[225] PARAM */) = sqrt(tmp2759);
  TRACE_POP
}

/*
equation index: 13553
type: SIMPLE_ASSIGN
r_init[224] = r_min + 224.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13553};
  (data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (224.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13554
type: SIMPLE_ASSIGN
omega_c[224] = sqrt(G * Md / (r_init[224] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13554(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13554};
  modelica_real tmp2760;
  modelica_real tmp2761;
  modelica_real tmp2762;
  modelica_real tmp2763;
  modelica_real tmp2764;
  modelica_real tmp2765;
  modelica_real tmp2766;
  modelica_real tmp2767;
  modelica_real tmp2768;
  modelica_real tmp2769;
  tmp2760 = (data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */);
  tmp2761 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2762 = (tmp2760 * tmp2760) + (tmp2761 * tmp2761);
  tmp2763 = 1.5;
  if(tmp2762 < 0.0 && tmp2763 != 0.0)
  {
    tmp2765 = modf(tmp2763, &tmp2766);
    
    if(tmp2765 > 0.5)
    {
      tmp2765 -= 1.0;
      tmp2766 += 1.0;
    }
    else if(tmp2765 < -0.5)
    {
      tmp2765 += 1.0;
      tmp2766 -= 1.0;
    }
    
    if(fabs(tmp2765) < 1e-10)
      tmp2764 = pow(tmp2762, tmp2766);
    else
    {
      tmp2768 = modf(1.0/tmp2763, &tmp2767);
      if(tmp2768 > 0.5)
      {
        tmp2768 -= 1.0;
        tmp2767 += 1.0;
      }
      else if(tmp2768 < -0.5)
      {
        tmp2768 += 1.0;
        tmp2767 -= 1.0;
      }
      if(fabs(tmp2768) < 1e-10 && ((unsigned long)tmp2767 & 1))
      {
        tmp2764 = -pow(-tmp2762, tmp2765)*pow(tmp2762, tmp2766);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2762, tmp2763);
      }
    }
  }
  else
  {
    tmp2764 = pow(tmp2762, tmp2763);
  }
  if(isnan(tmp2764) || isinf(tmp2764))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2762, tmp2763);
  }tmp2769 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2764,"(r_init[224] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2769 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[224] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2769);
    }
  }
  (data->simulationInfo->realParameter[728] /* omega_c[224] PARAM */) = sqrt(tmp2769);
  TRACE_POP
}

/*
equation index: 13555
type: SIMPLE_ASSIGN
r_init[223] = r_min + 223.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13555};
  (data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (223.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13556
type: SIMPLE_ASSIGN
omega_c[223] = sqrt(G * Md / (r_init[223] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13556};
  modelica_real tmp2770;
  modelica_real tmp2771;
  modelica_real tmp2772;
  modelica_real tmp2773;
  modelica_real tmp2774;
  modelica_real tmp2775;
  modelica_real tmp2776;
  modelica_real tmp2777;
  modelica_real tmp2778;
  modelica_real tmp2779;
  tmp2770 = (data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */);
  tmp2771 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2772 = (tmp2770 * tmp2770) + (tmp2771 * tmp2771);
  tmp2773 = 1.5;
  if(tmp2772 < 0.0 && tmp2773 != 0.0)
  {
    tmp2775 = modf(tmp2773, &tmp2776);
    
    if(tmp2775 > 0.5)
    {
      tmp2775 -= 1.0;
      tmp2776 += 1.0;
    }
    else if(tmp2775 < -0.5)
    {
      tmp2775 += 1.0;
      tmp2776 -= 1.0;
    }
    
    if(fabs(tmp2775) < 1e-10)
      tmp2774 = pow(tmp2772, tmp2776);
    else
    {
      tmp2778 = modf(1.0/tmp2773, &tmp2777);
      if(tmp2778 > 0.5)
      {
        tmp2778 -= 1.0;
        tmp2777 += 1.0;
      }
      else if(tmp2778 < -0.5)
      {
        tmp2778 += 1.0;
        tmp2777 -= 1.0;
      }
      if(fabs(tmp2778) < 1e-10 && ((unsigned long)tmp2777 & 1))
      {
        tmp2774 = -pow(-tmp2772, tmp2775)*pow(tmp2772, tmp2776);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2772, tmp2773);
      }
    }
  }
  else
  {
    tmp2774 = pow(tmp2772, tmp2773);
  }
  if(isnan(tmp2774) || isinf(tmp2774))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2772, tmp2773);
  }tmp2779 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2774,"(r_init[223] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2779 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[223] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2779);
    }
  }
  (data->simulationInfo->realParameter[727] /* omega_c[223] PARAM */) = sqrt(tmp2779);
  TRACE_POP
}

/*
equation index: 13557
type: SIMPLE_ASSIGN
r_init[222] = r_min + 222.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13557};
  (data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (222.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13558
type: SIMPLE_ASSIGN
omega_c[222] = sqrt(G * Md / (r_init[222] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13558(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13558};
  modelica_real tmp2780;
  modelica_real tmp2781;
  modelica_real tmp2782;
  modelica_real tmp2783;
  modelica_real tmp2784;
  modelica_real tmp2785;
  modelica_real tmp2786;
  modelica_real tmp2787;
  modelica_real tmp2788;
  modelica_real tmp2789;
  tmp2780 = (data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */);
  tmp2781 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2782 = (tmp2780 * tmp2780) + (tmp2781 * tmp2781);
  tmp2783 = 1.5;
  if(tmp2782 < 0.0 && tmp2783 != 0.0)
  {
    tmp2785 = modf(tmp2783, &tmp2786);
    
    if(tmp2785 > 0.5)
    {
      tmp2785 -= 1.0;
      tmp2786 += 1.0;
    }
    else if(tmp2785 < -0.5)
    {
      tmp2785 += 1.0;
      tmp2786 -= 1.0;
    }
    
    if(fabs(tmp2785) < 1e-10)
      tmp2784 = pow(tmp2782, tmp2786);
    else
    {
      tmp2788 = modf(1.0/tmp2783, &tmp2787);
      if(tmp2788 > 0.5)
      {
        tmp2788 -= 1.0;
        tmp2787 += 1.0;
      }
      else if(tmp2788 < -0.5)
      {
        tmp2788 += 1.0;
        tmp2787 -= 1.0;
      }
      if(fabs(tmp2788) < 1e-10 && ((unsigned long)tmp2787 & 1))
      {
        tmp2784 = -pow(-tmp2782, tmp2785)*pow(tmp2782, tmp2786);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2782, tmp2783);
      }
    }
  }
  else
  {
    tmp2784 = pow(tmp2782, tmp2783);
  }
  if(isnan(tmp2784) || isinf(tmp2784))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2782, tmp2783);
  }tmp2789 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2784,"(r_init[222] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2789 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[222] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2789);
    }
  }
  (data->simulationInfo->realParameter[726] /* omega_c[222] PARAM */) = sqrt(tmp2789);
  TRACE_POP
}

/*
equation index: 13559
type: SIMPLE_ASSIGN
r_init[221] = r_min + 221.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13559};
  (data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (221.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13560
type: SIMPLE_ASSIGN
omega_c[221] = sqrt(G * Md / (r_init[221] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13560(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13560};
  modelica_real tmp2790;
  modelica_real tmp2791;
  modelica_real tmp2792;
  modelica_real tmp2793;
  modelica_real tmp2794;
  modelica_real tmp2795;
  modelica_real tmp2796;
  modelica_real tmp2797;
  modelica_real tmp2798;
  modelica_real tmp2799;
  tmp2790 = (data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */);
  tmp2791 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2792 = (tmp2790 * tmp2790) + (tmp2791 * tmp2791);
  tmp2793 = 1.5;
  if(tmp2792 < 0.0 && tmp2793 != 0.0)
  {
    tmp2795 = modf(tmp2793, &tmp2796);
    
    if(tmp2795 > 0.5)
    {
      tmp2795 -= 1.0;
      tmp2796 += 1.0;
    }
    else if(tmp2795 < -0.5)
    {
      tmp2795 += 1.0;
      tmp2796 -= 1.0;
    }
    
    if(fabs(tmp2795) < 1e-10)
      tmp2794 = pow(tmp2792, tmp2796);
    else
    {
      tmp2798 = modf(1.0/tmp2793, &tmp2797);
      if(tmp2798 > 0.5)
      {
        tmp2798 -= 1.0;
        tmp2797 += 1.0;
      }
      else if(tmp2798 < -0.5)
      {
        tmp2798 += 1.0;
        tmp2797 -= 1.0;
      }
      if(fabs(tmp2798) < 1e-10 && ((unsigned long)tmp2797 & 1))
      {
        tmp2794 = -pow(-tmp2792, tmp2795)*pow(tmp2792, tmp2796);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2792, tmp2793);
      }
    }
  }
  else
  {
    tmp2794 = pow(tmp2792, tmp2793);
  }
  if(isnan(tmp2794) || isinf(tmp2794))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2792, tmp2793);
  }tmp2799 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2794,"(r_init[221] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2799 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[221] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2799);
    }
  }
  (data->simulationInfo->realParameter[725] /* omega_c[221] PARAM */) = sqrt(tmp2799);
  TRACE_POP
}

/*
equation index: 13561
type: SIMPLE_ASSIGN
r_init[220] = r_min + 220.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13561};
  (data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (220.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13562
type: SIMPLE_ASSIGN
omega_c[220] = sqrt(G * Md / (r_init[220] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13562(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13562};
  modelica_real tmp2800;
  modelica_real tmp2801;
  modelica_real tmp2802;
  modelica_real tmp2803;
  modelica_real tmp2804;
  modelica_real tmp2805;
  modelica_real tmp2806;
  modelica_real tmp2807;
  modelica_real tmp2808;
  modelica_real tmp2809;
  tmp2800 = (data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */);
  tmp2801 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2802 = (tmp2800 * tmp2800) + (tmp2801 * tmp2801);
  tmp2803 = 1.5;
  if(tmp2802 < 0.0 && tmp2803 != 0.0)
  {
    tmp2805 = modf(tmp2803, &tmp2806);
    
    if(tmp2805 > 0.5)
    {
      tmp2805 -= 1.0;
      tmp2806 += 1.0;
    }
    else if(tmp2805 < -0.5)
    {
      tmp2805 += 1.0;
      tmp2806 -= 1.0;
    }
    
    if(fabs(tmp2805) < 1e-10)
      tmp2804 = pow(tmp2802, tmp2806);
    else
    {
      tmp2808 = modf(1.0/tmp2803, &tmp2807);
      if(tmp2808 > 0.5)
      {
        tmp2808 -= 1.0;
        tmp2807 += 1.0;
      }
      else if(tmp2808 < -0.5)
      {
        tmp2808 += 1.0;
        tmp2807 -= 1.0;
      }
      if(fabs(tmp2808) < 1e-10 && ((unsigned long)tmp2807 & 1))
      {
        tmp2804 = -pow(-tmp2802, tmp2805)*pow(tmp2802, tmp2806);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2802, tmp2803);
      }
    }
  }
  else
  {
    tmp2804 = pow(tmp2802, tmp2803);
  }
  if(isnan(tmp2804) || isinf(tmp2804))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2802, tmp2803);
  }tmp2809 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2804,"(r_init[220] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2809 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[220] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2809);
    }
  }
  (data->simulationInfo->realParameter[724] /* omega_c[220] PARAM */) = sqrt(tmp2809);
  TRACE_POP
}

/*
equation index: 13563
type: SIMPLE_ASSIGN
r_init[219] = r_min + 219.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13563};
  (data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (219.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13564
type: SIMPLE_ASSIGN
omega_c[219] = sqrt(G * Md / (r_init[219] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13564};
  modelica_real tmp2810;
  modelica_real tmp2811;
  modelica_real tmp2812;
  modelica_real tmp2813;
  modelica_real tmp2814;
  modelica_real tmp2815;
  modelica_real tmp2816;
  modelica_real tmp2817;
  modelica_real tmp2818;
  modelica_real tmp2819;
  tmp2810 = (data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */);
  tmp2811 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2812 = (tmp2810 * tmp2810) + (tmp2811 * tmp2811);
  tmp2813 = 1.5;
  if(tmp2812 < 0.0 && tmp2813 != 0.0)
  {
    tmp2815 = modf(tmp2813, &tmp2816);
    
    if(tmp2815 > 0.5)
    {
      tmp2815 -= 1.0;
      tmp2816 += 1.0;
    }
    else if(tmp2815 < -0.5)
    {
      tmp2815 += 1.0;
      tmp2816 -= 1.0;
    }
    
    if(fabs(tmp2815) < 1e-10)
      tmp2814 = pow(tmp2812, tmp2816);
    else
    {
      tmp2818 = modf(1.0/tmp2813, &tmp2817);
      if(tmp2818 > 0.5)
      {
        tmp2818 -= 1.0;
        tmp2817 += 1.0;
      }
      else if(tmp2818 < -0.5)
      {
        tmp2818 += 1.0;
        tmp2817 -= 1.0;
      }
      if(fabs(tmp2818) < 1e-10 && ((unsigned long)tmp2817 & 1))
      {
        tmp2814 = -pow(-tmp2812, tmp2815)*pow(tmp2812, tmp2816);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2812, tmp2813);
      }
    }
  }
  else
  {
    tmp2814 = pow(tmp2812, tmp2813);
  }
  if(isnan(tmp2814) || isinf(tmp2814))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2812, tmp2813);
  }tmp2819 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2814,"(r_init[219] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2819 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[219] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2819);
    }
  }
  (data->simulationInfo->realParameter[723] /* omega_c[219] PARAM */) = sqrt(tmp2819);
  TRACE_POP
}

/*
equation index: 13565
type: SIMPLE_ASSIGN
r_init[218] = r_min + 218.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13565};
  (data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (218.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13566
type: SIMPLE_ASSIGN
omega_c[218] = sqrt(G * Md / (r_init[218] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13566(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13566};
  modelica_real tmp2820;
  modelica_real tmp2821;
  modelica_real tmp2822;
  modelica_real tmp2823;
  modelica_real tmp2824;
  modelica_real tmp2825;
  modelica_real tmp2826;
  modelica_real tmp2827;
  modelica_real tmp2828;
  modelica_real tmp2829;
  tmp2820 = (data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */);
  tmp2821 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2822 = (tmp2820 * tmp2820) + (tmp2821 * tmp2821);
  tmp2823 = 1.5;
  if(tmp2822 < 0.0 && tmp2823 != 0.0)
  {
    tmp2825 = modf(tmp2823, &tmp2826);
    
    if(tmp2825 > 0.5)
    {
      tmp2825 -= 1.0;
      tmp2826 += 1.0;
    }
    else if(tmp2825 < -0.5)
    {
      tmp2825 += 1.0;
      tmp2826 -= 1.0;
    }
    
    if(fabs(tmp2825) < 1e-10)
      tmp2824 = pow(tmp2822, tmp2826);
    else
    {
      tmp2828 = modf(1.0/tmp2823, &tmp2827);
      if(tmp2828 > 0.5)
      {
        tmp2828 -= 1.0;
        tmp2827 += 1.0;
      }
      else if(tmp2828 < -0.5)
      {
        tmp2828 += 1.0;
        tmp2827 -= 1.0;
      }
      if(fabs(tmp2828) < 1e-10 && ((unsigned long)tmp2827 & 1))
      {
        tmp2824 = -pow(-tmp2822, tmp2825)*pow(tmp2822, tmp2826);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2822, tmp2823);
      }
    }
  }
  else
  {
    tmp2824 = pow(tmp2822, tmp2823);
  }
  if(isnan(tmp2824) || isinf(tmp2824))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2822, tmp2823);
  }tmp2829 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2824,"(r_init[218] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2829 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[218] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2829);
    }
  }
  (data->simulationInfo->realParameter[722] /* omega_c[218] PARAM */) = sqrt(tmp2829);
  TRACE_POP
}

/*
equation index: 13567
type: SIMPLE_ASSIGN
r_init[217] = r_min + 217.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13567};
  (data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (217.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13568
type: SIMPLE_ASSIGN
omega_c[217] = sqrt(G * Md / (r_init[217] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13568(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13568};
  modelica_real tmp2830;
  modelica_real tmp2831;
  modelica_real tmp2832;
  modelica_real tmp2833;
  modelica_real tmp2834;
  modelica_real tmp2835;
  modelica_real tmp2836;
  modelica_real tmp2837;
  modelica_real tmp2838;
  modelica_real tmp2839;
  tmp2830 = (data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */);
  tmp2831 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2832 = (tmp2830 * tmp2830) + (tmp2831 * tmp2831);
  tmp2833 = 1.5;
  if(tmp2832 < 0.0 && tmp2833 != 0.0)
  {
    tmp2835 = modf(tmp2833, &tmp2836);
    
    if(tmp2835 > 0.5)
    {
      tmp2835 -= 1.0;
      tmp2836 += 1.0;
    }
    else if(tmp2835 < -0.5)
    {
      tmp2835 += 1.0;
      tmp2836 -= 1.0;
    }
    
    if(fabs(tmp2835) < 1e-10)
      tmp2834 = pow(tmp2832, tmp2836);
    else
    {
      tmp2838 = modf(1.0/tmp2833, &tmp2837);
      if(tmp2838 > 0.5)
      {
        tmp2838 -= 1.0;
        tmp2837 += 1.0;
      }
      else if(tmp2838 < -0.5)
      {
        tmp2838 += 1.0;
        tmp2837 -= 1.0;
      }
      if(fabs(tmp2838) < 1e-10 && ((unsigned long)tmp2837 & 1))
      {
        tmp2834 = -pow(-tmp2832, tmp2835)*pow(tmp2832, tmp2836);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2832, tmp2833);
      }
    }
  }
  else
  {
    tmp2834 = pow(tmp2832, tmp2833);
  }
  if(isnan(tmp2834) || isinf(tmp2834))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2832, tmp2833);
  }tmp2839 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2834,"(r_init[217] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2839 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[217] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2839);
    }
  }
  (data->simulationInfo->realParameter[721] /* omega_c[217] PARAM */) = sqrt(tmp2839);
  TRACE_POP
}

/*
equation index: 13569
type: SIMPLE_ASSIGN
r_init[216] = r_min + 216.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13569};
  (data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (216.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13570
type: SIMPLE_ASSIGN
omega_c[216] = sqrt(G * Md / (r_init[216] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13570(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13570};
  modelica_real tmp2840;
  modelica_real tmp2841;
  modelica_real tmp2842;
  modelica_real tmp2843;
  modelica_real tmp2844;
  modelica_real tmp2845;
  modelica_real tmp2846;
  modelica_real tmp2847;
  modelica_real tmp2848;
  modelica_real tmp2849;
  tmp2840 = (data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */);
  tmp2841 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2842 = (tmp2840 * tmp2840) + (tmp2841 * tmp2841);
  tmp2843 = 1.5;
  if(tmp2842 < 0.0 && tmp2843 != 0.0)
  {
    tmp2845 = modf(tmp2843, &tmp2846);
    
    if(tmp2845 > 0.5)
    {
      tmp2845 -= 1.0;
      tmp2846 += 1.0;
    }
    else if(tmp2845 < -0.5)
    {
      tmp2845 += 1.0;
      tmp2846 -= 1.0;
    }
    
    if(fabs(tmp2845) < 1e-10)
      tmp2844 = pow(tmp2842, tmp2846);
    else
    {
      tmp2848 = modf(1.0/tmp2843, &tmp2847);
      if(tmp2848 > 0.5)
      {
        tmp2848 -= 1.0;
        tmp2847 += 1.0;
      }
      else if(tmp2848 < -0.5)
      {
        tmp2848 += 1.0;
        tmp2847 -= 1.0;
      }
      if(fabs(tmp2848) < 1e-10 && ((unsigned long)tmp2847 & 1))
      {
        tmp2844 = -pow(-tmp2842, tmp2845)*pow(tmp2842, tmp2846);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2842, tmp2843);
      }
    }
  }
  else
  {
    tmp2844 = pow(tmp2842, tmp2843);
  }
  if(isnan(tmp2844) || isinf(tmp2844))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2842, tmp2843);
  }tmp2849 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2844,"(r_init[216] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2849 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[216] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2849);
    }
  }
  (data->simulationInfo->realParameter[720] /* omega_c[216] PARAM */) = sqrt(tmp2849);
  TRACE_POP
}

/*
equation index: 13571
type: SIMPLE_ASSIGN
r_init[215] = r_min + 215.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13571};
  (data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (215.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13572
type: SIMPLE_ASSIGN
omega_c[215] = sqrt(G * Md / (r_init[215] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13572};
  modelica_real tmp2850;
  modelica_real tmp2851;
  modelica_real tmp2852;
  modelica_real tmp2853;
  modelica_real tmp2854;
  modelica_real tmp2855;
  modelica_real tmp2856;
  modelica_real tmp2857;
  modelica_real tmp2858;
  modelica_real tmp2859;
  tmp2850 = (data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */);
  tmp2851 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2852 = (tmp2850 * tmp2850) + (tmp2851 * tmp2851);
  tmp2853 = 1.5;
  if(tmp2852 < 0.0 && tmp2853 != 0.0)
  {
    tmp2855 = modf(tmp2853, &tmp2856);
    
    if(tmp2855 > 0.5)
    {
      tmp2855 -= 1.0;
      tmp2856 += 1.0;
    }
    else if(tmp2855 < -0.5)
    {
      tmp2855 += 1.0;
      tmp2856 -= 1.0;
    }
    
    if(fabs(tmp2855) < 1e-10)
      tmp2854 = pow(tmp2852, tmp2856);
    else
    {
      tmp2858 = modf(1.0/tmp2853, &tmp2857);
      if(tmp2858 > 0.5)
      {
        tmp2858 -= 1.0;
        tmp2857 += 1.0;
      }
      else if(tmp2858 < -0.5)
      {
        tmp2858 += 1.0;
        tmp2857 -= 1.0;
      }
      if(fabs(tmp2858) < 1e-10 && ((unsigned long)tmp2857 & 1))
      {
        tmp2854 = -pow(-tmp2852, tmp2855)*pow(tmp2852, tmp2856);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2852, tmp2853);
      }
    }
  }
  else
  {
    tmp2854 = pow(tmp2852, tmp2853);
  }
  if(isnan(tmp2854) || isinf(tmp2854))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2852, tmp2853);
  }tmp2859 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2854,"(r_init[215] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2859 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[215] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2859);
    }
  }
  (data->simulationInfo->realParameter[719] /* omega_c[215] PARAM */) = sqrt(tmp2859);
  TRACE_POP
}

/*
equation index: 13573
type: SIMPLE_ASSIGN
r_init[214] = r_min + 214.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13573};
  (data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (214.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13574
type: SIMPLE_ASSIGN
omega_c[214] = sqrt(G * Md / (r_init[214] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13574(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13574};
  modelica_real tmp2860;
  modelica_real tmp2861;
  modelica_real tmp2862;
  modelica_real tmp2863;
  modelica_real tmp2864;
  modelica_real tmp2865;
  modelica_real tmp2866;
  modelica_real tmp2867;
  modelica_real tmp2868;
  modelica_real tmp2869;
  tmp2860 = (data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */);
  tmp2861 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2862 = (tmp2860 * tmp2860) + (tmp2861 * tmp2861);
  tmp2863 = 1.5;
  if(tmp2862 < 0.0 && tmp2863 != 0.0)
  {
    tmp2865 = modf(tmp2863, &tmp2866);
    
    if(tmp2865 > 0.5)
    {
      tmp2865 -= 1.0;
      tmp2866 += 1.0;
    }
    else if(tmp2865 < -0.5)
    {
      tmp2865 += 1.0;
      tmp2866 -= 1.0;
    }
    
    if(fabs(tmp2865) < 1e-10)
      tmp2864 = pow(tmp2862, tmp2866);
    else
    {
      tmp2868 = modf(1.0/tmp2863, &tmp2867);
      if(tmp2868 > 0.5)
      {
        tmp2868 -= 1.0;
        tmp2867 += 1.0;
      }
      else if(tmp2868 < -0.5)
      {
        tmp2868 += 1.0;
        tmp2867 -= 1.0;
      }
      if(fabs(tmp2868) < 1e-10 && ((unsigned long)tmp2867 & 1))
      {
        tmp2864 = -pow(-tmp2862, tmp2865)*pow(tmp2862, tmp2866);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2862, tmp2863);
      }
    }
  }
  else
  {
    tmp2864 = pow(tmp2862, tmp2863);
  }
  if(isnan(tmp2864) || isinf(tmp2864))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2862, tmp2863);
  }tmp2869 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2864,"(r_init[214] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2869 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[214] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2869);
    }
  }
  (data->simulationInfo->realParameter[718] /* omega_c[214] PARAM */) = sqrt(tmp2869);
  TRACE_POP
}

/*
equation index: 13575
type: SIMPLE_ASSIGN
r_init[213] = r_min + 213.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13575};
  (data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (213.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13576
type: SIMPLE_ASSIGN
omega_c[213] = sqrt(G * Md / (r_init[213] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13576(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13576};
  modelica_real tmp2870;
  modelica_real tmp2871;
  modelica_real tmp2872;
  modelica_real tmp2873;
  modelica_real tmp2874;
  modelica_real tmp2875;
  modelica_real tmp2876;
  modelica_real tmp2877;
  modelica_real tmp2878;
  modelica_real tmp2879;
  tmp2870 = (data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */);
  tmp2871 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2872 = (tmp2870 * tmp2870) + (tmp2871 * tmp2871);
  tmp2873 = 1.5;
  if(tmp2872 < 0.0 && tmp2873 != 0.0)
  {
    tmp2875 = modf(tmp2873, &tmp2876);
    
    if(tmp2875 > 0.5)
    {
      tmp2875 -= 1.0;
      tmp2876 += 1.0;
    }
    else if(tmp2875 < -0.5)
    {
      tmp2875 += 1.0;
      tmp2876 -= 1.0;
    }
    
    if(fabs(tmp2875) < 1e-10)
      tmp2874 = pow(tmp2872, tmp2876);
    else
    {
      tmp2878 = modf(1.0/tmp2873, &tmp2877);
      if(tmp2878 > 0.5)
      {
        tmp2878 -= 1.0;
        tmp2877 += 1.0;
      }
      else if(tmp2878 < -0.5)
      {
        tmp2878 += 1.0;
        tmp2877 -= 1.0;
      }
      if(fabs(tmp2878) < 1e-10 && ((unsigned long)tmp2877 & 1))
      {
        tmp2874 = -pow(-tmp2872, tmp2875)*pow(tmp2872, tmp2876);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2872, tmp2873);
      }
    }
  }
  else
  {
    tmp2874 = pow(tmp2872, tmp2873);
  }
  if(isnan(tmp2874) || isinf(tmp2874))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2872, tmp2873);
  }tmp2879 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2874,"(r_init[213] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2879 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[213] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2879);
    }
  }
  (data->simulationInfo->realParameter[717] /* omega_c[213] PARAM */) = sqrt(tmp2879);
  TRACE_POP
}

/*
equation index: 13577
type: SIMPLE_ASSIGN
r_init[212] = r_min + 212.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13577};
  (data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (212.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13578
type: SIMPLE_ASSIGN
omega_c[212] = sqrt(G * Md / (r_init[212] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13578(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13578};
  modelica_real tmp2880;
  modelica_real tmp2881;
  modelica_real tmp2882;
  modelica_real tmp2883;
  modelica_real tmp2884;
  modelica_real tmp2885;
  modelica_real tmp2886;
  modelica_real tmp2887;
  modelica_real tmp2888;
  modelica_real tmp2889;
  tmp2880 = (data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */);
  tmp2881 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2882 = (tmp2880 * tmp2880) + (tmp2881 * tmp2881);
  tmp2883 = 1.5;
  if(tmp2882 < 0.0 && tmp2883 != 0.0)
  {
    tmp2885 = modf(tmp2883, &tmp2886);
    
    if(tmp2885 > 0.5)
    {
      tmp2885 -= 1.0;
      tmp2886 += 1.0;
    }
    else if(tmp2885 < -0.5)
    {
      tmp2885 += 1.0;
      tmp2886 -= 1.0;
    }
    
    if(fabs(tmp2885) < 1e-10)
      tmp2884 = pow(tmp2882, tmp2886);
    else
    {
      tmp2888 = modf(1.0/tmp2883, &tmp2887);
      if(tmp2888 > 0.5)
      {
        tmp2888 -= 1.0;
        tmp2887 += 1.0;
      }
      else if(tmp2888 < -0.5)
      {
        tmp2888 += 1.0;
        tmp2887 -= 1.0;
      }
      if(fabs(tmp2888) < 1e-10 && ((unsigned long)tmp2887 & 1))
      {
        tmp2884 = -pow(-tmp2882, tmp2885)*pow(tmp2882, tmp2886);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2882, tmp2883);
      }
    }
  }
  else
  {
    tmp2884 = pow(tmp2882, tmp2883);
  }
  if(isnan(tmp2884) || isinf(tmp2884))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2882, tmp2883);
  }tmp2889 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2884,"(r_init[212] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2889 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[212] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2889);
    }
  }
  (data->simulationInfo->realParameter[716] /* omega_c[212] PARAM */) = sqrt(tmp2889);
  TRACE_POP
}

/*
equation index: 13579
type: SIMPLE_ASSIGN
r_init[211] = r_min + 211.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13579};
  (data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (211.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13580
type: SIMPLE_ASSIGN
omega_c[211] = sqrt(G * Md / (r_init[211] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13580};
  modelica_real tmp2890;
  modelica_real tmp2891;
  modelica_real tmp2892;
  modelica_real tmp2893;
  modelica_real tmp2894;
  modelica_real tmp2895;
  modelica_real tmp2896;
  modelica_real tmp2897;
  modelica_real tmp2898;
  modelica_real tmp2899;
  tmp2890 = (data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */);
  tmp2891 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2892 = (tmp2890 * tmp2890) + (tmp2891 * tmp2891);
  tmp2893 = 1.5;
  if(tmp2892 < 0.0 && tmp2893 != 0.0)
  {
    tmp2895 = modf(tmp2893, &tmp2896);
    
    if(tmp2895 > 0.5)
    {
      tmp2895 -= 1.0;
      tmp2896 += 1.0;
    }
    else if(tmp2895 < -0.5)
    {
      tmp2895 += 1.0;
      tmp2896 -= 1.0;
    }
    
    if(fabs(tmp2895) < 1e-10)
      tmp2894 = pow(tmp2892, tmp2896);
    else
    {
      tmp2898 = modf(1.0/tmp2893, &tmp2897);
      if(tmp2898 > 0.5)
      {
        tmp2898 -= 1.0;
        tmp2897 += 1.0;
      }
      else if(tmp2898 < -0.5)
      {
        tmp2898 += 1.0;
        tmp2897 -= 1.0;
      }
      if(fabs(tmp2898) < 1e-10 && ((unsigned long)tmp2897 & 1))
      {
        tmp2894 = -pow(-tmp2892, tmp2895)*pow(tmp2892, tmp2896);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2892, tmp2893);
      }
    }
  }
  else
  {
    tmp2894 = pow(tmp2892, tmp2893);
  }
  if(isnan(tmp2894) || isinf(tmp2894))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2892, tmp2893);
  }tmp2899 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2894,"(r_init[211] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2899 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[211] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2899);
    }
  }
  (data->simulationInfo->realParameter[715] /* omega_c[211] PARAM */) = sqrt(tmp2899);
  TRACE_POP
}

/*
equation index: 13581
type: SIMPLE_ASSIGN
r_init[210] = r_min + 210.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13581};
  (data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (210.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13582
type: SIMPLE_ASSIGN
omega_c[210] = sqrt(G * Md / (r_init[210] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13582(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13582};
  modelica_real tmp2900;
  modelica_real tmp2901;
  modelica_real tmp2902;
  modelica_real tmp2903;
  modelica_real tmp2904;
  modelica_real tmp2905;
  modelica_real tmp2906;
  modelica_real tmp2907;
  modelica_real tmp2908;
  modelica_real tmp2909;
  tmp2900 = (data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */);
  tmp2901 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2902 = (tmp2900 * tmp2900) + (tmp2901 * tmp2901);
  tmp2903 = 1.5;
  if(tmp2902 < 0.0 && tmp2903 != 0.0)
  {
    tmp2905 = modf(tmp2903, &tmp2906);
    
    if(tmp2905 > 0.5)
    {
      tmp2905 -= 1.0;
      tmp2906 += 1.0;
    }
    else if(tmp2905 < -0.5)
    {
      tmp2905 += 1.0;
      tmp2906 -= 1.0;
    }
    
    if(fabs(tmp2905) < 1e-10)
      tmp2904 = pow(tmp2902, tmp2906);
    else
    {
      tmp2908 = modf(1.0/tmp2903, &tmp2907);
      if(tmp2908 > 0.5)
      {
        tmp2908 -= 1.0;
        tmp2907 += 1.0;
      }
      else if(tmp2908 < -0.5)
      {
        tmp2908 += 1.0;
        tmp2907 -= 1.0;
      }
      if(fabs(tmp2908) < 1e-10 && ((unsigned long)tmp2907 & 1))
      {
        tmp2904 = -pow(-tmp2902, tmp2905)*pow(tmp2902, tmp2906);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2902, tmp2903);
      }
    }
  }
  else
  {
    tmp2904 = pow(tmp2902, tmp2903);
  }
  if(isnan(tmp2904) || isinf(tmp2904))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2902, tmp2903);
  }tmp2909 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2904,"(r_init[210] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2909 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[210] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2909);
    }
  }
  (data->simulationInfo->realParameter[714] /* omega_c[210] PARAM */) = sqrt(tmp2909);
  TRACE_POP
}

/*
equation index: 13583
type: SIMPLE_ASSIGN
r_init[209] = r_min + 209.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13583};
  (data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (209.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13584
type: SIMPLE_ASSIGN
omega_c[209] = sqrt(G * Md / (r_init[209] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13584(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13584};
  modelica_real tmp2910;
  modelica_real tmp2911;
  modelica_real tmp2912;
  modelica_real tmp2913;
  modelica_real tmp2914;
  modelica_real tmp2915;
  modelica_real tmp2916;
  modelica_real tmp2917;
  modelica_real tmp2918;
  modelica_real tmp2919;
  tmp2910 = (data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */);
  tmp2911 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2912 = (tmp2910 * tmp2910) + (tmp2911 * tmp2911);
  tmp2913 = 1.5;
  if(tmp2912 < 0.0 && tmp2913 != 0.0)
  {
    tmp2915 = modf(tmp2913, &tmp2916);
    
    if(tmp2915 > 0.5)
    {
      tmp2915 -= 1.0;
      tmp2916 += 1.0;
    }
    else if(tmp2915 < -0.5)
    {
      tmp2915 += 1.0;
      tmp2916 -= 1.0;
    }
    
    if(fabs(tmp2915) < 1e-10)
      tmp2914 = pow(tmp2912, tmp2916);
    else
    {
      tmp2918 = modf(1.0/tmp2913, &tmp2917);
      if(tmp2918 > 0.5)
      {
        tmp2918 -= 1.0;
        tmp2917 += 1.0;
      }
      else if(tmp2918 < -0.5)
      {
        tmp2918 += 1.0;
        tmp2917 -= 1.0;
      }
      if(fabs(tmp2918) < 1e-10 && ((unsigned long)tmp2917 & 1))
      {
        tmp2914 = -pow(-tmp2912, tmp2915)*pow(tmp2912, tmp2916);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2912, tmp2913);
      }
    }
  }
  else
  {
    tmp2914 = pow(tmp2912, tmp2913);
  }
  if(isnan(tmp2914) || isinf(tmp2914))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2912, tmp2913);
  }tmp2919 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2914,"(r_init[209] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2919 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[209] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2919);
    }
  }
  (data->simulationInfo->realParameter[713] /* omega_c[209] PARAM */) = sqrt(tmp2919);
  TRACE_POP
}

/*
equation index: 13585
type: SIMPLE_ASSIGN
r_init[208] = r_min + 208.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13585};
  (data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (208.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13586
type: SIMPLE_ASSIGN
omega_c[208] = sqrt(G * Md / (r_init[208] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13586(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13586};
  modelica_real tmp2920;
  modelica_real tmp2921;
  modelica_real tmp2922;
  modelica_real tmp2923;
  modelica_real tmp2924;
  modelica_real tmp2925;
  modelica_real tmp2926;
  modelica_real tmp2927;
  modelica_real tmp2928;
  modelica_real tmp2929;
  tmp2920 = (data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */);
  tmp2921 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2922 = (tmp2920 * tmp2920) + (tmp2921 * tmp2921);
  tmp2923 = 1.5;
  if(tmp2922 < 0.0 && tmp2923 != 0.0)
  {
    tmp2925 = modf(tmp2923, &tmp2926);
    
    if(tmp2925 > 0.5)
    {
      tmp2925 -= 1.0;
      tmp2926 += 1.0;
    }
    else if(tmp2925 < -0.5)
    {
      tmp2925 += 1.0;
      tmp2926 -= 1.0;
    }
    
    if(fabs(tmp2925) < 1e-10)
      tmp2924 = pow(tmp2922, tmp2926);
    else
    {
      tmp2928 = modf(1.0/tmp2923, &tmp2927);
      if(tmp2928 > 0.5)
      {
        tmp2928 -= 1.0;
        tmp2927 += 1.0;
      }
      else if(tmp2928 < -0.5)
      {
        tmp2928 += 1.0;
        tmp2927 -= 1.0;
      }
      if(fabs(tmp2928) < 1e-10 && ((unsigned long)tmp2927 & 1))
      {
        tmp2924 = -pow(-tmp2922, tmp2925)*pow(tmp2922, tmp2926);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2922, tmp2923);
      }
    }
  }
  else
  {
    tmp2924 = pow(tmp2922, tmp2923);
  }
  if(isnan(tmp2924) || isinf(tmp2924))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2922, tmp2923);
  }tmp2929 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2924,"(r_init[208] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2929 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[208] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2929);
    }
  }
  (data->simulationInfo->realParameter[712] /* omega_c[208] PARAM */) = sqrt(tmp2929);
  TRACE_POP
}

/*
equation index: 13587
type: SIMPLE_ASSIGN
r_init[207] = r_min + 207.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13587};
  (data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (207.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13588
type: SIMPLE_ASSIGN
omega_c[207] = sqrt(G * Md / (r_init[207] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13588};
  modelica_real tmp2930;
  modelica_real tmp2931;
  modelica_real tmp2932;
  modelica_real tmp2933;
  modelica_real tmp2934;
  modelica_real tmp2935;
  modelica_real tmp2936;
  modelica_real tmp2937;
  modelica_real tmp2938;
  modelica_real tmp2939;
  tmp2930 = (data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */);
  tmp2931 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2932 = (tmp2930 * tmp2930) + (tmp2931 * tmp2931);
  tmp2933 = 1.5;
  if(tmp2932 < 0.0 && tmp2933 != 0.0)
  {
    tmp2935 = modf(tmp2933, &tmp2936);
    
    if(tmp2935 > 0.5)
    {
      tmp2935 -= 1.0;
      tmp2936 += 1.0;
    }
    else if(tmp2935 < -0.5)
    {
      tmp2935 += 1.0;
      tmp2936 -= 1.0;
    }
    
    if(fabs(tmp2935) < 1e-10)
      tmp2934 = pow(tmp2932, tmp2936);
    else
    {
      tmp2938 = modf(1.0/tmp2933, &tmp2937);
      if(tmp2938 > 0.5)
      {
        tmp2938 -= 1.0;
        tmp2937 += 1.0;
      }
      else if(tmp2938 < -0.5)
      {
        tmp2938 += 1.0;
        tmp2937 -= 1.0;
      }
      if(fabs(tmp2938) < 1e-10 && ((unsigned long)tmp2937 & 1))
      {
        tmp2934 = -pow(-tmp2932, tmp2935)*pow(tmp2932, tmp2936);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2932, tmp2933);
      }
    }
  }
  else
  {
    tmp2934 = pow(tmp2932, tmp2933);
  }
  if(isnan(tmp2934) || isinf(tmp2934))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2932, tmp2933);
  }tmp2939 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2934,"(r_init[207] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2939 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[207] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2939);
    }
  }
  (data->simulationInfo->realParameter[711] /* omega_c[207] PARAM */) = sqrt(tmp2939);
  TRACE_POP
}

/*
equation index: 13589
type: SIMPLE_ASSIGN
r_init[206] = r_min + 206.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13589};
  (data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (206.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13590
type: SIMPLE_ASSIGN
omega_c[206] = sqrt(G * Md / (r_init[206] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13590(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13590};
  modelica_real tmp2940;
  modelica_real tmp2941;
  modelica_real tmp2942;
  modelica_real tmp2943;
  modelica_real tmp2944;
  modelica_real tmp2945;
  modelica_real tmp2946;
  modelica_real tmp2947;
  modelica_real tmp2948;
  modelica_real tmp2949;
  tmp2940 = (data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */);
  tmp2941 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2942 = (tmp2940 * tmp2940) + (tmp2941 * tmp2941);
  tmp2943 = 1.5;
  if(tmp2942 < 0.0 && tmp2943 != 0.0)
  {
    tmp2945 = modf(tmp2943, &tmp2946);
    
    if(tmp2945 > 0.5)
    {
      tmp2945 -= 1.0;
      tmp2946 += 1.0;
    }
    else if(tmp2945 < -0.5)
    {
      tmp2945 += 1.0;
      tmp2946 -= 1.0;
    }
    
    if(fabs(tmp2945) < 1e-10)
      tmp2944 = pow(tmp2942, tmp2946);
    else
    {
      tmp2948 = modf(1.0/tmp2943, &tmp2947);
      if(tmp2948 > 0.5)
      {
        tmp2948 -= 1.0;
        tmp2947 += 1.0;
      }
      else if(tmp2948 < -0.5)
      {
        tmp2948 += 1.0;
        tmp2947 -= 1.0;
      }
      if(fabs(tmp2948) < 1e-10 && ((unsigned long)tmp2947 & 1))
      {
        tmp2944 = -pow(-tmp2942, tmp2945)*pow(tmp2942, tmp2946);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2942, tmp2943);
      }
    }
  }
  else
  {
    tmp2944 = pow(tmp2942, tmp2943);
  }
  if(isnan(tmp2944) || isinf(tmp2944))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2942, tmp2943);
  }tmp2949 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2944,"(r_init[206] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2949 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[206] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2949);
    }
  }
  (data->simulationInfo->realParameter[710] /* omega_c[206] PARAM */) = sqrt(tmp2949);
  TRACE_POP
}

/*
equation index: 13591
type: SIMPLE_ASSIGN
r_init[205] = r_min + 205.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13591};
  (data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (205.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13592
type: SIMPLE_ASSIGN
omega_c[205] = sqrt(G * Md / (r_init[205] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13592(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13592};
  modelica_real tmp2950;
  modelica_real tmp2951;
  modelica_real tmp2952;
  modelica_real tmp2953;
  modelica_real tmp2954;
  modelica_real tmp2955;
  modelica_real tmp2956;
  modelica_real tmp2957;
  modelica_real tmp2958;
  modelica_real tmp2959;
  tmp2950 = (data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */);
  tmp2951 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2952 = (tmp2950 * tmp2950) + (tmp2951 * tmp2951);
  tmp2953 = 1.5;
  if(tmp2952 < 0.0 && tmp2953 != 0.0)
  {
    tmp2955 = modf(tmp2953, &tmp2956);
    
    if(tmp2955 > 0.5)
    {
      tmp2955 -= 1.0;
      tmp2956 += 1.0;
    }
    else if(tmp2955 < -0.5)
    {
      tmp2955 += 1.0;
      tmp2956 -= 1.0;
    }
    
    if(fabs(tmp2955) < 1e-10)
      tmp2954 = pow(tmp2952, tmp2956);
    else
    {
      tmp2958 = modf(1.0/tmp2953, &tmp2957);
      if(tmp2958 > 0.5)
      {
        tmp2958 -= 1.0;
        tmp2957 += 1.0;
      }
      else if(tmp2958 < -0.5)
      {
        tmp2958 += 1.0;
        tmp2957 -= 1.0;
      }
      if(fabs(tmp2958) < 1e-10 && ((unsigned long)tmp2957 & 1))
      {
        tmp2954 = -pow(-tmp2952, tmp2955)*pow(tmp2952, tmp2956);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2952, tmp2953);
      }
    }
  }
  else
  {
    tmp2954 = pow(tmp2952, tmp2953);
  }
  if(isnan(tmp2954) || isinf(tmp2954))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2952, tmp2953);
  }tmp2959 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2954,"(r_init[205] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2959 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[205] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2959);
    }
  }
  (data->simulationInfo->realParameter[709] /* omega_c[205] PARAM */) = sqrt(tmp2959);
  TRACE_POP
}

/*
equation index: 13593
type: SIMPLE_ASSIGN
r_init[204] = r_min + 204.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13593};
  (data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (204.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13594
type: SIMPLE_ASSIGN
omega_c[204] = sqrt(G * Md / (r_init[204] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13594(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13594};
  modelica_real tmp2960;
  modelica_real tmp2961;
  modelica_real tmp2962;
  modelica_real tmp2963;
  modelica_real tmp2964;
  modelica_real tmp2965;
  modelica_real tmp2966;
  modelica_real tmp2967;
  modelica_real tmp2968;
  modelica_real tmp2969;
  tmp2960 = (data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */);
  tmp2961 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2962 = (tmp2960 * tmp2960) + (tmp2961 * tmp2961);
  tmp2963 = 1.5;
  if(tmp2962 < 0.0 && tmp2963 != 0.0)
  {
    tmp2965 = modf(tmp2963, &tmp2966);
    
    if(tmp2965 > 0.5)
    {
      tmp2965 -= 1.0;
      tmp2966 += 1.0;
    }
    else if(tmp2965 < -0.5)
    {
      tmp2965 += 1.0;
      tmp2966 -= 1.0;
    }
    
    if(fabs(tmp2965) < 1e-10)
      tmp2964 = pow(tmp2962, tmp2966);
    else
    {
      tmp2968 = modf(1.0/tmp2963, &tmp2967);
      if(tmp2968 > 0.5)
      {
        tmp2968 -= 1.0;
        tmp2967 += 1.0;
      }
      else if(tmp2968 < -0.5)
      {
        tmp2968 += 1.0;
        tmp2967 -= 1.0;
      }
      if(fabs(tmp2968) < 1e-10 && ((unsigned long)tmp2967 & 1))
      {
        tmp2964 = -pow(-tmp2962, tmp2965)*pow(tmp2962, tmp2966);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2962, tmp2963);
      }
    }
  }
  else
  {
    tmp2964 = pow(tmp2962, tmp2963);
  }
  if(isnan(tmp2964) || isinf(tmp2964))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2962, tmp2963);
  }tmp2969 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2964,"(r_init[204] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2969 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[204] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2969);
    }
  }
  (data->simulationInfo->realParameter[708] /* omega_c[204] PARAM */) = sqrt(tmp2969);
  TRACE_POP
}

/*
equation index: 13595
type: SIMPLE_ASSIGN
r_init[203] = r_min + 203.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13595};
  (data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (203.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13596
type: SIMPLE_ASSIGN
omega_c[203] = sqrt(G * Md / (r_init[203] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13596};
  modelica_real tmp2970;
  modelica_real tmp2971;
  modelica_real tmp2972;
  modelica_real tmp2973;
  modelica_real tmp2974;
  modelica_real tmp2975;
  modelica_real tmp2976;
  modelica_real tmp2977;
  modelica_real tmp2978;
  modelica_real tmp2979;
  tmp2970 = (data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */);
  tmp2971 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2972 = (tmp2970 * tmp2970) + (tmp2971 * tmp2971);
  tmp2973 = 1.5;
  if(tmp2972 < 0.0 && tmp2973 != 0.0)
  {
    tmp2975 = modf(tmp2973, &tmp2976);
    
    if(tmp2975 > 0.5)
    {
      tmp2975 -= 1.0;
      tmp2976 += 1.0;
    }
    else if(tmp2975 < -0.5)
    {
      tmp2975 += 1.0;
      tmp2976 -= 1.0;
    }
    
    if(fabs(tmp2975) < 1e-10)
      tmp2974 = pow(tmp2972, tmp2976);
    else
    {
      tmp2978 = modf(1.0/tmp2973, &tmp2977);
      if(tmp2978 > 0.5)
      {
        tmp2978 -= 1.0;
        tmp2977 += 1.0;
      }
      else if(tmp2978 < -0.5)
      {
        tmp2978 += 1.0;
        tmp2977 -= 1.0;
      }
      if(fabs(tmp2978) < 1e-10 && ((unsigned long)tmp2977 & 1))
      {
        tmp2974 = -pow(-tmp2972, tmp2975)*pow(tmp2972, tmp2976);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2972, tmp2973);
      }
    }
  }
  else
  {
    tmp2974 = pow(tmp2972, tmp2973);
  }
  if(isnan(tmp2974) || isinf(tmp2974))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2972, tmp2973);
  }tmp2979 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2974,"(r_init[203] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2979 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[203] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2979);
    }
  }
  (data->simulationInfo->realParameter[707] /* omega_c[203] PARAM */) = sqrt(tmp2979);
  TRACE_POP
}

/*
equation index: 13597
type: SIMPLE_ASSIGN
r_init[202] = r_min + 202.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13597};
  (data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (202.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13598
type: SIMPLE_ASSIGN
omega_c[202] = sqrt(G * Md / (r_init[202] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13598(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13598};
  modelica_real tmp2980;
  modelica_real tmp2981;
  modelica_real tmp2982;
  modelica_real tmp2983;
  modelica_real tmp2984;
  modelica_real tmp2985;
  modelica_real tmp2986;
  modelica_real tmp2987;
  modelica_real tmp2988;
  modelica_real tmp2989;
  tmp2980 = (data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */);
  tmp2981 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2982 = (tmp2980 * tmp2980) + (tmp2981 * tmp2981);
  tmp2983 = 1.5;
  if(tmp2982 < 0.0 && tmp2983 != 0.0)
  {
    tmp2985 = modf(tmp2983, &tmp2986);
    
    if(tmp2985 > 0.5)
    {
      tmp2985 -= 1.0;
      tmp2986 += 1.0;
    }
    else if(tmp2985 < -0.5)
    {
      tmp2985 += 1.0;
      tmp2986 -= 1.0;
    }
    
    if(fabs(tmp2985) < 1e-10)
      tmp2984 = pow(tmp2982, tmp2986);
    else
    {
      tmp2988 = modf(1.0/tmp2983, &tmp2987);
      if(tmp2988 > 0.5)
      {
        tmp2988 -= 1.0;
        tmp2987 += 1.0;
      }
      else if(tmp2988 < -0.5)
      {
        tmp2988 += 1.0;
        tmp2987 -= 1.0;
      }
      if(fabs(tmp2988) < 1e-10 && ((unsigned long)tmp2987 & 1))
      {
        tmp2984 = -pow(-tmp2982, tmp2985)*pow(tmp2982, tmp2986);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2982, tmp2983);
      }
    }
  }
  else
  {
    tmp2984 = pow(tmp2982, tmp2983);
  }
  if(isnan(tmp2984) || isinf(tmp2984))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2982, tmp2983);
  }tmp2989 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2984,"(r_init[202] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2989 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[202] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2989);
    }
  }
  (data->simulationInfo->realParameter[706] /* omega_c[202] PARAM */) = sqrt(tmp2989);
  TRACE_POP
}

/*
equation index: 13599
type: SIMPLE_ASSIGN
r_init[201] = r_min + 201.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13599};
  (data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (201.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13600
type: SIMPLE_ASSIGN
omega_c[201] = sqrt(G * Md / (r_init[201] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13600(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13600};
  modelica_real tmp2990;
  modelica_real tmp2991;
  modelica_real tmp2992;
  modelica_real tmp2993;
  modelica_real tmp2994;
  modelica_real tmp2995;
  modelica_real tmp2996;
  modelica_real tmp2997;
  modelica_real tmp2998;
  modelica_real tmp2999;
  tmp2990 = (data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */);
  tmp2991 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2992 = (tmp2990 * tmp2990) + (tmp2991 * tmp2991);
  tmp2993 = 1.5;
  if(tmp2992 < 0.0 && tmp2993 != 0.0)
  {
    tmp2995 = modf(tmp2993, &tmp2996);
    
    if(tmp2995 > 0.5)
    {
      tmp2995 -= 1.0;
      tmp2996 += 1.0;
    }
    else if(tmp2995 < -0.5)
    {
      tmp2995 += 1.0;
      tmp2996 -= 1.0;
    }
    
    if(fabs(tmp2995) < 1e-10)
      tmp2994 = pow(tmp2992, tmp2996);
    else
    {
      tmp2998 = modf(1.0/tmp2993, &tmp2997);
      if(tmp2998 > 0.5)
      {
        tmp2998 -= 1.0;
        tmp2997 += 1.0;
      }
      else if(tmp2998 < -0.5)
      {
        tmp2998 += 1.0;
        tmp2997 -= 1.0;
      }
      if(fabs(tmp2998) < 1e-10 && ((unsigned long)tmp2997 & 1))
      {
        tmp2994 = -pow(-tmp2992, tmp2995)*pow(tmp2992, tmp2996);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2992, tmp2993);
      }
    }
  }
  else
  {
    tmp2994 = pow(tmp2992, tmp2993);
  }
  if(isnan(tmp2994) || isinf(tmp2994))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2992, tmp2993);
  }tmp2999 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2994,"(r_init[201] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2999 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[201] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2999);
    }
  }
  (data->simulationInfo->realParameter[705] /* omega_c[201] PARAM */) = sqrt(tmp2999);
  TRACE_POP
}

/*
equation index: 13601
type: SIMPLE_ASSIGN
r_init[200] = r_min + 200.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13601};
  (data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (200.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13602
type: SIMPLE_ASSIGN
omega_c[200] = sqrt(G * Md / (r_init[200] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13602(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13602};
  modelica_real tmp3000;
  modelica_real tmp3001;
  modelica_real tmp3002;
  modelica_real tmp3003;
  modelica_real tmp3004;
  modelica_real tmp3005;
  modelica_real tmp3006;
  modelica_real tmp3007;
  modelica_real tmp3008;
  modelica_real tmp3009;
  tmp3000 = (data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */);
  tmp3001 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3002 = (tmp3000 * tmp3000) + (tmp3001 * tmp3001);
  tmp3003 = 1.5;
  if(tmp3002 < 0.0 && tmp3003 != 0.0)
  {
    tmp3005 = modf(tmp3003, &tmp3006);
    
    if(tmp3005 > 0.5)
    {
      tmp3005 -= 1.0;
      tmp3006 += 1.0;
    }
    else if(tmp3005 < -0.5)
    {
      tmp3005 += 1.0;
      tmp3006 -= 1.0;
    }
    
    if(fabs(tmp3005) < 1e-10)
      tmp3004 = pow(tmp3002, tmp3006);
    else
    {
      tmp3008 = modf(1.0/tmp3003, &tmp3007);
      if(tmp3008 > 0.5)
      {
        tmp3008 -= 1.0;
        tmp3007 += 1.0;
      }
      else if(tmp3008 < -0.5)
      {
        tmp3008 += 1.0;
        tmp3007 -= 1.0;
      }
      if(fabs(tmp3008) < 1e-10 && ((unsigned long)tmp3007 & 1))
      {
        tmp3004 = -pow(-tmp3002, tmp3005)*pow(tmp3002, tmp3006);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3002, tmp3003);
      }
    }
  }
  else
  {
    tmp3004 = pow(tmp3002, tmp3003);
  }
  if(isnan(tmp3004) || isinf(tmp3004))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3002, tmp3003);
  }tmp3009 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3004,"(r_init[200] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3009 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[200] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3009);
    }
  }
  (data->simulationInfo->realParameter[704] /* omega_c[200] PARAM */) = sqrt(tmp3009);
  TRACE_POP
}

/*
equation index: 13603
type: SIMPLE_ASSIGN
r_init[199] = r_min + 199.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13603};
  (data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (199.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13604
type: SIMPLE_ASSIGN
omega_c[199] = sqrt(G * Md / (r_init[199] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13604};
  modelica_real tmp3010;
  modelica_real tmp3011;
  modelica_real tmp3012;
  modelica_real tmp3013;
  modelica_real tmp3014;
  modelica_real tmp3015;
  modelica_real tmp3016;
  modelica_real tmp3017;
  modelica_real tmp3018;
  modelica_real tmp3019;
  tmp3010 = (data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */);
  tmp3011 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3012 = (tmp3010 * tmp3010) + (tmp3011 * tmp3011);
  tmp3013 = 1.5;
  if(tmp3012 < 0.0 && tmp3013 != 0.0)
  {
    tmp3015 = modf(tmp3013, &tmp3016);
    
    if(tmp3015 > 0.5)
    {
      tmp3015 -= 1.0;
      tmp3016 += 1.0;
    }
    else if(tmp3015 < -0.5)
    {
      tmp3015 += 1.0;
      tmp3016 -= 1.0;
    }
    
    if(fabs(tmp3015) < 1e-10)
      tmp3014 = pow(tmp3012, tmp3016);
    else
    {
      tmp3018 = modf(1.0/tmp3013, &tmp3017);
      if(tmp3018 > 0.5)
      {
        tmp3018 -= 1.0;
        tmp3017 += 1.0;
      }
      else if(tmp3018 < -0.5)
      {
        tmp3018 += 1.0;
        tmp3017 -= 1.0;
      }
      if(fabs(tmp3018) < 1e-10 && ((unsigned long)tmp3017 & 1))
      {
        tmp3014 = -pow(-tmp3012, tmp3015)*pow(tmp3012, tmp3016);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3012, tmp3013);
      }
    }
  }
  else
  {
    tmp3014 = pow(tmp3012, tmp3013);
  }
  if(isnan(tmp3014) || isinf(tmp3014))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3012, tmp3013);
  }tmp3019 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3014,"(r_init[199] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3019 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[199] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3019);
    }
  }
  (data->simulationInfo->realParameter[703] /* omega_c[199] PARAM */) = sqrt(tmp3019);
  TRACE_POP
}

/*
equation index: 13605
type: SIMPLE_ASSIGN
r_init[198] = r_min + 198.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13605};
  (data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (198.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13606
type: SIMPLE_ASSIGN
omega_c[198] = sqrt(G * Md / (r_init[198] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13606(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13606};
  modelica_real tmp3020;
  modelica_real tmp3021;
  modelica_real tmp3022;
  modelica_real tmp3023;
  modelica_real tmp3024;
  modelica_real tmp3025;
  modelica_real tmp3026;
  modelica_real tmp3027;
  modelica_real tmp3028;
  modelica_real tmp3029;
  tmp3020 = (data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */);
  tmp3021 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3022 = (tmp3020 * tmp3020) + (tmp3021 * tmp3021);
  tmp3023 = 1.5;
  if(tmp3022 < 0.0 && tmp3023 != 0.0)
  {
    tmp3025 = modf(tmp3023, &tmp3026);
    
    if(tmp3025 > 0.5)
    {
      tmp3025 -= 1.0;
      tmp3026 += 1.0;
    }
    else if(tmp3025 < -0.5)
    {
      tmp3025 += 1.0;
      tmp3026 -= 1.0;
    }
    
    if(fabs(tmp3025) < 1e-10)
      tmp3024 = pow(tmp3022, tmp3026);
    else
    {
      tmp3028 = modf(1.0/tmp3023, &tmp3027);
      if(tmp3028 > 0.5)
      {
        tmp3028 -= 1.0;
        tmp3027 += 1.0;
      }
      else if(tmp3028 < -0.5)
      {
        tmp3028 += 1.0;
        tmp3027 -= 1.0;
      }
      if(fabs(tmp3028) < 1e-10 && ((unsigned long)tmp3027 & 1))
      {
        tmp3024 = -pow(-tmp3022, tmp3025)*pow(tmp3022, tmp3026);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3022, tmp3023);
      }
    }
  }
  else
  {
    tmp3024 = pow(tmp3022, tmp3023);
  }
  if(isnan(tmp3024) || isinf(tmp3024))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3022, tmp3023);
  }tmp3029 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3024,"(r_init[198] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3029 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[198] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3029);
    }
  }
  (data->simulationInfo->realParameter[702] /* omega_c[198] PARAM */) = sqrt(tmp3029);
  TRACE_POP
}

/*
equation index: 13607
type: SIMPLE_ASSIGN
r_init[197] = r_min + 197.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13607};
  (data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (197.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13608
type: SIMPLE_ASSIGN
omega_c[197] = sqrt(G * Md / (r_init[197] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13608(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13608};
  modelica_real tmp3030;
  modelica_real tmp3031;
  modelica_real tmp3032;
  modelica_real tmp3033;
  modelica_real tmp3034;
  modelica_real tmp3035;
  modelica_real tmp3036;
  modelica_real tmp3037;
  modelica_real tmp3038;
  modelica_real tmp3039;
  tmp3030 = (data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */);
  tmp3031 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3032 = (tmp3030 * tmp3030) + (tmp3031 * tmp3031);
  tmp3033 = 1.5;
  if(tmp3032 < 0.0 && tmp3033 != 0.0)
  {
    tmp3035 = modf(tmp3033, &tmp3036);
    
    if(tmp3035 > 0.5)
    {
      tmp3035 -= 1.0;
      tmp3036 += 1.0;
    }
    else if(tmp3035 < -0.5)
    {
      tmp3035 += 1.0;
      tmp3036 -= 1.0;
    }
    
    if(fabs(tmp3035) < 1e-10)
      tmp3034 = pow(tmp3032, tmp3036);
    else
    {
      tmp3038 = modf(1.0/tmp3033, &tmp3037);
      if(tmp3038 > 0.5)
      {
        tmp3038 -= 1.0;
        tmp3037 += 1.0;
      }
      else if(tmp3038 < -0.5)
      {
        tmp3038 += 1.0;
        tmp3037 -= 1.0;
      }
      if(fabs(tmp3038) < 1e-10 && ((unsigned long)tmp3037 & 1))
      {
        tmp3034 = -pow(-tmp3032, tmp3035)*pow(tmp3032, tmp3036);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3032, tmp3033);
      }
    }
  }
  else
  {
    tmp3034 = pow(tmp3032, tmp3033);
  }
  if(isnan(tmp3034) || isinf(tmp3034))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3032, tmp3033);
  }tmp3039 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3034,"(r_init[197] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3039 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[197] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3039);
    }
  }
  (data->simulationInfo->realParameter[701] /* omega_c[197] PARAM */) = sqrt(tmp3039);
  TRACE_POP
}

/*
equation index: 13609
type: SIMPLE_ASSIGN
r_init[196] = r_min + 196.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13609};
  (data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (196.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13610
type: SIMPLE_ASSIGN
omega_c[196] = sqrt(G * Md / (r_init[196] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13610(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13610};
  modelica_real tmp3040;
  modelica_real tmp3041;
  modelica_real tmp3042;
  modelica_real tmp3043;
  modelica_real tmp3044;
  modelica_real tmp3045;
  modelica_real tmp3046;
  modelica_real tmp3047;
  modelica_real tmp3048;
  modelica_real tmp3049;
  tmp3040 = (data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */);
  tmp3041 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3042 = (tmp3040 * tmp3040) + (tmp3041 * tmp3041);
  tmp3043 = 1.5;
  if(tmp3042 < 0.0 && tmp3043 != 0.0)
  {
    tmp3045 = modf(tmp3043, &tmp3046);
    
    if(tmp3045 > 0.5)
    {
      tmp3045 -= 1.0;
      tmp3046 += 1.0;
    }
    else if(tmp3045 < -0.5)
    {
      tmp3045 += 1.0;
      tmp3046 -= 1.0;
    }
    
    if(fabs(tmp3045) < 1e-10)
      tmp3044 = pow(tmp3042, tmp3046);
    else
    {
      tmp3048 = modf(1.0/tmp3043, &tmp3047);
      if(tmp3048 > 0.5)
      {
        tmp3048 -= 1.0;
        tmp3047 += 1.0;
      }
      else if(tmp3048 < -0.5)
      {
        tmp3048 += 1.0;
        tmp3047 -= 1.0;
      }
      if(fabs(tmp3048) < 1e-10 && ((unsigned long)tmp3047 & 1))
      {
        tmp3044 = -pow(-tmp3042, tmp3045)*pow(tmp3042, tmp3046);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3042, tmp3043);
      }
    }
  }
  else
  {
    tmp3044 = pow(tmp3042, tmp3043);
  }
  if(isnan(tmp3044) || isinf(tmp3044))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3042, tmp3043);
  }tmp3049 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3044,"(r_init[196] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3049 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[196] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3049);
    }
  }
  (data->simulationInfo->realParameter[700] /* omega_c[196] PARAM */) = sqrt(tmp3049);
  TRACE_POP
}

/*
equation index: 13611
type: SIMPLE_ASSIGN
r_init[195] = r_min + 195.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13611};
  (data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (195.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13612
type: SIMPLE_ASSIGN
omega_c[195] = sqrt(G * Md / (r_init[195] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13612};
  modelica_real tmp3050;
  modelica_real tmp3051;
  modelica_real tmp3052;
  modelica_real tmp3053;
  modelica_real tmp3054;
  modelica_real tmp3055;
  modelica_real tmp3056;
  modelica_real tmp3057;
  modelica_real tmp3058;
  modelica_real tmp3059;
  tmp3050 = (data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */);
  tmp3051 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3052 = (tmp3050 * tmp3050) + (tmp3051 * tmp3051);
  tmp3053 = 1.5;
  if(tmp3052 < 0.0 && tmp3053 != 0.0)
  {
    tmp3055 = modf(tmp3053, &tmp3056);
    
    if(tmp3055 > 0.5)
    {
      tmp3055 -= 1.0;
      tmp3056 += 1.0;
    }
    else if(tmp3055 < -0.5)
    {
      tmp3055 += 1.0;
      tmp3056 -= 1.0;
    }
    
    if(fabs(tmp3055) < 1e-10)
      tmp3054 = pow(tmp3052, tmp3056);
    else
    {
      tmp3058 = modf(1.0/tmp3053, &tmp3057);
      if(tmp3058 > 0.5)
      {
        tmp3058 -= 1.0;
        tmp3057 += 1.0;
      }
      else if(tmp3058 < -0.5)
      {
        tmp3058 += 1.0;
        tmp3057 -= 1.0;
      }
      if(fabs(tmp3058) < 1e-10 && ((unsigned long)tmp3057 & 1))
      {
        tmp3054 = -pow(-tmp3052, tmp3055)*pow(tmp3052, tmp3056);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3052, tmp3053);
      }
    }
  }
  else
  {
    tmp3054 = pow(tmp3052, tmp3053);
  }
  if(isnan(tmp3054) || isinf(tmp3054))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3052, tmp3053);
  }tmp3059 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3054,"(r_init[195] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3059 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[195] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3059);
    }
  }
  (data->simulationInfo->realParameter[699] /* omega_c[195] PARAM */) = sqrt(tmp3059);
  TRACE_POP
}

/*
equation index: 13613
type: SIMPLE_ASSIGN
r_init[194] = r_min + 194.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13613};
  (data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (194.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13614
type: SIMPLE_ASSIGN
omega_c[194] = sqrt(G * Md / (r_init[194] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13614(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13614};
  modelica_real tmp3060;
  modelica_real tmp3061;
  modelica_real tmp3062;
  modelica_real tmp3063;
  modelica_real tmp3064;
  modelica_real tmp3065;
  modelica_real tmp3066;
  modelica_real tmp3067;
  modelica_real tmp3068;
  modelica_real tmp3069;
  tmp3060 = (data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */);
  tmp3061 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3062 = (tmp3060 * tmp3060) + (tmp3061 * tmp3061);
  tmp3063 = 1.5;
  if(tmp3062 < 0.0 && tmp3063 != 0.0)
  {
    tmp3065 = modf(tmp3063, &tmp3066);
    
    if(tmp3065 > 0.5)
    {
      tmp3065 -= 1.0;
      tmp3066 += 1.0;
    }
    else if(tmp3065 < -0.5)
    {
      tmp3065 += 1.0;
      tmp3066 -= 1.0;
    }
    
    if(fabs(tmp3065) < 1e-10)
      tmp3064 = pow(tmp3062, tmp3066);
    else
    {
      tmp3068 = modf(1.0/tmp3063, &tmp3067);
      if(tmp3068 > 0.5)
      {
        tmp3068 -= 1.0;
        tmp3067 += 1.0;
      }
      else if(tmp3068 < -0.5)
      {
        tmp3068 += 1.0;
        tmp3067 -= 1.0;
      }
      if(fabs(tmp3068) < 1e-10 && ((unsigned long)tmp3067 & 1))
      {
        tmp3064 = -pow(-tmp3062, tmp3065)*pow(tmp3062, tmp3066);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3062, tmp3063);
      }
    }
  }
  else
  {
    tmp3064 = pow(tmp3062, tmp3063);
  }
  if(isnan(tmp3064) || isinf(tmp3064))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3062, tmp3063);
  }tmp3069 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3064,"(r_init[194] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3069 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[194] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3069);
    }
  }
  (data->simulationInfo->realParameter[698] /* omega_c[194] PARAM */) = sqrt(tmp3069);
  TRACE_POP
}

/*
equation index: 13615
type: SIMPLE_ASSIGN
r_init[193] = r_min + 193.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13615};
  (data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (193.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13616
type: SIMPLE_ASSIGN
omega_c[193] = sqrt(G * Md / (r_init[193] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13616(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13616};
  modelica_real tmp3070;
  modelica_real tmp3071;
  modelica_real tmp3072;
  modelica_real tmp3073;
  modelica_real tmp3074;
  modelica_real tmp3075;
  modelica_real tmp3076;
  modelica_real tmp3077;
  modelica_real tmp3078;
  modelica_real tmp3079;
  tmp3070 = (data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */);
  tmp3071 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3072 = (tmp3070 * tmp3070) + (tmp3071 * tmp3071);
  tmp3073 = 1.5;
  if(tmp3072 < 0.0 && tmp3073 != 0.0)
  {
    tmp3075 = modf(tmp3073, &tmp3076);
    
    if(tmp3075 > 0.5)
    {
      tmp3075 -= 1.0;
      tmp3076 += 1.0;
    }
    else if(tmp3075 < -0.5)
    {
      tmp3075 += 1.0;
      tmp3076 -= 1.0;
    }
    
    if(fabs(tmp3075) < 1e-10)
      tmp3074 = pow(tmp3072, tmp3076);
    else
    {
      tmp3078 = modf(1.0/tmp3073, &tmp3077);
      if(tmp3078 > 0.5)
      {
        tmp3078 -= 1.0;
        tmp3077 += 1.0;
      }
      else if(tmp3078 < -0.5)
      {
        tmp3078 += 1.0;
        tmp3077 -= 1.0;
      }
      if(fabs(tmp3078) < 1e-10 && ((unsigned long)tmp3077 & 1))
      {
        tmp3074 = -pow(-tmp3072, tmp3075)*pow(tmp3072, tmp3076);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3072, tmp3073);
      }
    }
  }
  else
  {
    tmp3074 = pow(tmp3072, tmp3073);
  }
  if(isnan(tmp3074) || isinf(tmp3074))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3072, tmp3073);
  }tmp3079 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3074,"(r_init[193] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3079 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[193] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3079);
    }
  }
  (data->simulationInfo->realParameter[697] /* omega_c[193] PARAM */) = sqrt(tmp3079);
  TRACE_POP
}

/*
equation index: 13617
type: SIMPLE_ASSIGN
r_init[192] = r_min + 192.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13617};
  (data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (192.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13618
type: SIMPLE_ASSIGN
omega_c[192] = sqrt(G * Md / (r_init[192] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13618(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13618};
  modelica_real tmp3080;
  modelica_real tmp3081;
  modelica_real tmp3082;
  modelica_real tmp3083;
  modelica_real tmp3084;
  modelica_real tmp3085;
  modelica_real tmp3086;
  modelica_real tmp3087;
  modelica_real tmp3088;
  modelica_real tmp3089;
  tmp3080 = (data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */);
  tmp3081 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3082 = (tmp3080 * tmp3080) + (tmp3081 * tmp3081);
  tmp3083 = 1.5;
  if(tmp3082 < 0.0 && tmp3083 != 0.0)
  {
    tmp3085 = modf(tmp3083, &tmp3086);
    
    if(tmp3085 > 0.5)
    {
      tmp3085 -= 1.0;
      tmp3086 += 1.0;
    }
    else if(tmp3085 < -0.5)
    {
      tmp3085 += 1.0;
      tmp3086 -= 1.0;
    }
    
    if(fabs(tmp3085) < 1e-10)
      tmp3084 = pow(tmp3082, tmp3086);
    else
    {
      tmp3088 = modf(1.0/tmp3083, &tmp3087);
      if(tmp3088 > 0.5)
      {
        tmp3088 -= 1.0;
        tmp3087 += 1.0;
      }
      else if(tmp3088 < -0.5)
      {
        tmp3088 += 1.0;
        tmp3087 -= 1.0;
      }
      if(fabs(tmp3088) < 1e-10 && ((unsigned long)tmp3087 & 1))
      {
        tmp3084 = -pow(-tmp3082, tmp3085)*pow(tmp3082, tmp3086);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3082, tmp3083);
      }
    }
  }
  else
  {
    tmp3084 = pow(tmp3082, tmp3083);
  }
  if(isnan(tmp3084) || isinf(tmp3084))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3082, tmp3083);
  }tmp3089 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3084,"(r_init[192] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3089 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[192] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3089);
    }
  }
  (data->simulationInfo->realParameter[696] /* omega_c[192] PARAM */) = sqrt(tmp3089);
  TRACE_POP
}

/*
equation index: 13619
type: SIMPLE_ASSIGN
r_init[191] = r_min + 191.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13619};
  (data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (191.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13620
type: SIMPLE_ASSIGN
omega_c[191] = sqrt(G * Md / (r_init[191] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13620};
  modelica_real tmp3090;
  modelica_real tmp3091;
  modelica_real tmp3092;
  modelica_real tmp3093;
  modelica_real tmp3094;
  modelica_real tmp3095;
  modelica_real tmp3096;
  modelica_real tmp3097;
  modelica_real tmp3098;
  modelica_real tmp3099;
  tmp3090 = (data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */);
  tmp3091 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3092 = (tmp3090 * tmp3090) + (tmp3091 * tmp3091);
  tmp3093 = 1.5;
  if(tmp3092 < 0.0 && tmp3093 != 0.0)
  {
    tmp3095 = modf(tmp3093, &tmp3096);
    
    if(tmp3095 > 0.5)
    {
      tmp3095 -= 1.0;
      tmp3096 += 1.0;
    }
    else if(tmp3095 < -0.5)
    {
      tmp3095 += 1.0;
      tmp3096 -= 1.0;
    }
    
    if(fabs(tmp3095) < 1e-10)
      tmp3094 = pow(tmp3092, tmp3096);
    else
    {
      tmp3098 = modf(1.0/tmp3093, &tmp3097);
      if(tmp3098 > 0.5)
      {
        tmp3098 -= 1.0;
        tmp3097 += 1.0;
      }
      else if(tmp3098 < -0.5)
      {
        tmp3098 += 1.0;
        tmp3097 -= 1.0;
      }
      if(fabs(tmp3098) < 1e-10 && ((unsigned long)tmp3097 & 1))
      {
        tmp3094 = -pow(-tmp3092, tmp3095)*pow(tmp3092, tmp3096);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3092, tmp3093);
      }
    }
  }
  else
  {
    tmp3094 = pow(tmp3092, tmp3093);
  }
  if(isnan(tmp3094) || isinf(tmp3094))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3092, tmp3093);
  }tmp3099 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3094,"(r_init[191] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3099 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[191] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3099);
    }
  }
  (data->simulationInfo->realParameter[695] /* omega_c[191] PARAM */) = sqrt(tmp3099);
  TRACE_POP
}

/*
equation index: 13621
type: SIMPLE_ASSIGN
r_init[190] = r_min + 190.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13621};
  (data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (190.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13622
type: SIMPLE_ASSIGN
omega_c[190] = sqrt(G * Md / (r_init[190] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13622(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13622};
  modelica_real tmp3100;
  modelica_real tmp3101;
  modelica_real tmp3102;
  modelica_real tmp3103;
  modelica_real tmp3104;
  modelica_real tmp3105;
  modelica_real tmp3106;
  modelica_real tmp3107;
  modelica_real tmp3108;
  modelica_real tmp3109;
  tmp3100 = (data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */);
  tmp3101 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3102 = (tmp3100 * tmp3100) + (tmp3101 * tmp3101);
  tmp3103 = 1.5;
  if(tmp3102 < 0.0 && tmp3103 != 0.0)
  {
    tmp3105 = modf(tmp3103, &tmp3106);
    
    if(tmp3105 > 0.5)
    {
      tmp3105 -= 1.0;
      tmp3106 += 1.0;
    }
    else if(tmp3105 < -0.5)
    {
      tmp3105 += 1.0;
      tmp3106 -= 1.0;
    }
    
    if(fabs(tmp3105) < 1e-10)
      tmp3104 = pow(tmp3102, tmp3106);
    else
    {
      tmp3108 = modf(1.0/tmp3103, &tmp3107);
      if(tmp3108 > 0.5)
      {
        tmp3108 -= 1.0;
        tmp3107 += 1.0;
      }
      else if(tmp3108 < -0.5)
      {
        tmp3108 += 1.0;
        tmp3107 -= 1.0;
      }
      if(fabs(tmp3108) < 1e-10 && ((unsigned long)tmp3107 & 1))
      {
        tmp3104 = -pow(-tmp3102, tmp3105)*pow(tmp3102, tmp3106);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3102, tmp3103);
      }
    }
  }
  else
  {
    tmp3104 = pow(tmp3102, tmp3103);
  }
  if(isnan(tmp3104) || isinf(tmp3104))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3102, tmp3103);
  }tmp3109 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3104,"(r_init[190] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3109 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[190] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3109);
    }
  }
  (data->simulationInfo->realParameter[694] /* omega_c[190] PARAM */) = sqrt(tmp3109);
  TRACE_POP
}

/*
equation index: 13623
type: SIMPLE_ASSIGN
r_init[189] = r_min + 189.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13623};
  (data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (189.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13624
type: SIMPLE_ASSIGN
omega_c[189] = sqrt(G * Md / (r_init[189] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13624(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13624};
  modelica_real tmp3110;
  modelica_real tmp3111;
  modelica_real tmp3112;
  modelica_real tmp3113;
  modelica_real tmp3114;
  modelica_real tmp3115;
  modelica_real tmp3116;
  modelica_real tmp3117;
  modelica_real tmp3118;
  modelica_real tmp3119;
  tmp3110 = (data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */);
  tmp3111 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3112 = (tmp3110 * tmp3110) + (tmp3111 * tmp3111);
  tmp3113 = 1.5;
  if(tmp3112 < 0.0 && tmp3113 != 0.0)
  {
    tmp3115 = modf(tmp3113, &tmp3116);
    
    if(tmp3115 > 0.5)
    {
      tmp3115 -= 1.0;
      tmp3116 += 1.0;
    }
    else if(tmp3115 < -0.5)
    {
      tmp3115 += 1.0;
      tmp3116 -= 1.0;
    }
    
    if(fabs(tmp3115) < 1e-10)
      tmp3114 = pow(tmp3112, tmp3116);
    else
    {
      tmp3118 = modf(1.0/tmp3113, &tmp3117);
      if(tmp3118 > 0.5)
      {
        tmp3118 -= 1.0;
        tmp3117 += 1.0;
      }
      else if(tmp3118 < -0.5)
      {
        tmp3118 += 1.0;
        tmp3117 -= 1.0;
      }
      if(fabs(tmp3118) < 1e-10 && ((unsigned long)tmp3117 & 1))
      {
        tmp3114 = -pow(-tmp3112, tmp3115)*pow(tmp3112, tmp3116);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3112, tmp3113);
      }
    }
  }
  else
  {
    tmp3114 = pow(tmp3112, tmp3113);
  }
  if(isnan(tmp3114) || isinf(tmp3114))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3112, tmp3113);
  }tmp3119 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3114,"(r_init[189] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3119 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[189] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3119);
    }
  }
  (data->simulationInfo->realParameter[693] /* omega_c[189] PARAM */) = sqrt(tmp3119);
  TRACE_POP
}

/*
equation index: 13625
type: SIMPLE_ASSIGN
r_init[188] = r_min + 188.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13625};
  (data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (188.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13626
type: SIMPLE_ASSIGN
omega_c[188] = sqrt(G * Md / (r_init[188] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13626(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13626};
  modelica_real tmp3120;
  modelica_real tmp3121;
  modelica_real tmp3122;
  modelica_real tmp3123;
  modelica_real tmp3124;
  modelica_real tmp3125;
  modelica_real tmp3126;
  modelica_real tmp3127;
  modelica_real tmp3128;
  modelica_real tmp3129;
  tmp3120 = (data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */);
  tmp3121 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3122 = (tmp3120 * tmp3120) + (tmp3121 * tmp3121);
  tmp3123 = 1.5;
  if(tmp3122 < 0.0 && tmp3123 != 0.0)
  {
    tmp3125 = modf(tmp3123, &tmp3126);
    
    if(tmp3125 > 0.5)
    {
      tmp3125 -= 1.0;
      tmp3126 += 1.0;
    }
    else if(tmp3125 < -0.5)
    {
      tmp3125 += 1.0;
      tmp3126 -= 1.0;
    }
    
    if(fabs(tmp3125) < 1e-10)
      tmp3124 = pow(tmp3122, tmp3126);
    else
    {
      tmp3128 = modf(1.0/tmp3123, &tmp3127);
      if(tmp3128 > 0.5)
      {
        tmp3128 -= 1.0;
        tmp3127 += 1.0;
      }
      else if(tmp3128 < -0.5)
      {
        tmp3128 += 1.0;
        tmp3127 -= 1.0;
      }
      if(fabs(tmp3128) < 1e-10 && ((unsigned long)tmp3127 & 1))
      {
        tmp3124 = -pow(-tmp3122, tmp3125)*pow(tmp3122, tmp3126);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3122, tmp3123);
      }
    }
  }
  else
  {
    tmp3124 = pow(tmp3122, tmp3123);
  }
  if(isnan(tmp3124) || isinf(tmp3124))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3122, tmp3123);
  }tmp3129 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3124,"(r_init[188] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3129 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[188] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3129);
    }
  }
  (data->simulationInfo->realParameter[692] /* omega_c[188] PARAM */) = sqrt(tmp3129);
  TRACE_POP
}

/*
equation index: 13627
type: SIMPLE_ASSIGN
r_init[187] = r_min + 187.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13627};
  (data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (187.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13628
type: SIMPLE_ASSIGN
omega_c[187] = sqrt(G * Md / (r_init[187] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13628};
  modelica_real tmp3130;
  modelica_real tmp3131;
  modelica_real tmp3132;
  modelica_real tmp3133;
  modelica_real tmp3134;
  modelica_real tmp3135;
  modelica_real tmp3136;
  modelica_real tmp3137;
  modelica_real tmp3138;
  modelica_real tmp3139;
  tmp3130 = (data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */);
  tmp3131 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3132 = (tmp3130 * tmp3130) + (tmp3131 * tmp3131);
  tmp3133 = 1.5;
  if(tmp3132 < 0.0 && tmp3133 != 0.0)
  {
    tmp3135 = modf(tmp3133, &tmp3136);
    
    if(tmp3135 > 0.5)
    {
      tmp3135 -= 1.0;
      tmp3136 += 1.0;
    }
    else if(tmp3135 < -0.5)
    {
      tmp3135 += 1.0;
      tmp3136 -= 1.0;
    }
    
    if(fabs(tmp3135) < 1e-10)
      tmp3134 = pow(tmp3132, tmp3136);
    else
    {
      tmp3138 = modf(1.0/tmp3133, &tmp3137);
      if(tmp3138 > 0.5)
      {
        tmp3138 -= 1.0;
        tmp3137 += 1.0;
      }
      else if(tmp3138 < -0.5)
      {
        tmp3138 += 1.0;
        tmp3137 -= 1.0;
      }
      if(fabs(tmp3138) < 1e-10 && ((unsigned long)tmp3137 & 1))
      {
        tmp3134 = -pow(-tmp3132, tmp3135)*pow(tmp3132, tmp3136);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3132, tmp3133);
      }
    }
  }
  else
  {
    tmp3134 = pow(tmp3132, tmp3133);
  }
  if(isnan(tmp3134) || isinf(tmp3134))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3132, tmp3133);
  }tmp3139 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3134,"(r_init[187] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3139 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[187] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3139);
    }
  }
  (data->simulationInfo->realParameter[691] /* omega_c[187] PARAM */) = sqrt(tmp3139);
  TRACE_POP
}

/*
equation index: 13629
type: SIMPLE_ASSIGN
r_init[186] = r_min + 186.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13629};
  (data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (186.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13630
type: SIMPLE_ASSIGN
omega_c[186] = sqrt(G * Md / (r_init[186] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13630(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13630};
  modelica_real tmp3140;
  modelica_real tmp3141;
  modelica_real tmp3142;
  modelica_real tmp3143;
  modelica_real tmp3144;
  modelica_real tmp3145;
  modelica_real tmp3146;
  modelica_real tmp3147;
  modelica_real tmp3148;
  modelica_real tmp3149;
  tmp3140 = (data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */);
  tmp3141 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3142 = (tmp3140 * tmp3140) + (tmp3141 * tmp3141);
  tmp3143 = 1.5;
  if(tmp3142 < 0.0 && tmp3143 != 0.0)
  {
    tmp3145 = modf(tmp3143, &tmp3146);
    
    if(tmp3145 > 0.5)
    {
      tmp3145 -= 1.0;
      tmp3146 += 1.0;
    }
    else if(tmp3145 < -0.5)
    {
      tmp3145 += 1.0;
      tmp3146 -= 1.0;
    }
    
    if(fabs(tmp3145) < 1e-10)
      tmp3144 = pow(tmp3142, tmp3146);
    else
    {
      tmp3148 = modf(1.0/tmp3143, &tmp3147);
      if(tmp3148 > 0.5)
      {
        tmp3148 -= 1.0;
        tmp3147 += 1.0;
      }
      else if(tmp3148 < -0.5)
      {
        tmp3148 += 1.0;
        tmp3147 -= 1.0;
      }
      if(fabs(tmp3148) < 1e-10 && ((unsigned long)tmp3147 & 1))
      {
        tmp3144 = -pow(-tmp3142, tmp3145)*pow(tmp3142, tmp3146);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3142, tmp3143);
      }
    }
  }
  else
  {
    tmp3144 = pow(tmp3142, tmp3143);
  }
  if(isnan(tmp3144) || isinf(tmp3144))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3142, tmp3143);
  }tmp3149 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3144,"(r_init[186] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3149 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[186] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3149);
    }
  }
  (data->simulationInfo->realParameter[690] /* omega_c[186] PARAM */) = sqrt(tmp3149);
  TRACE_POP
}

/*
equation index: 13631
type: SIMPLE_ASSIGN
r_init[185] = r_min + 185.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13631};
  (data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (185.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13632
type: SIMPLE_ASSIGN
omega_c[185] = sqrt(G * Md / (r_init[185] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13632(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13632};
  modelica_real tmp3150;
  modelica_real tmp3151;
  modelica_real tmp3152;
  modelica_real tmp3153;
  modelica_real tmp3154;
  modelica_real tmp3155;
  modelica_real tmp3156;
  modelica_real tmp3157;
  modelica_real tmp3158;
  modelica_real tmp3159;
  tmp3150 = (data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */);
  tmp3151 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3152 = (tmp3150 * tmp3150) + (tmp3151 * tmp3151);
  tmp3153 = 1.5;
  if(tmp3152 < 0.0 && tmp3153 != 0.0)
  {
    tmp3155 = modf(tmp3153, &tmp3156);
    
    if(tmp3155 > 0.5)
    {
      tmp3155 -= 1.0;
      tmp3156 += 1.0;
    }
    else if(tmp3155 < -0.5)
    {
      tmp3155 += 1.0;
      tmp3156 -= 1.0;
    }
    
    if(fabs(tmp3155) < 1e-10)
      tmp3154 = pow(tmp3152, tmp3156);
    else
    {
      tmp3158 = modf(1.0/tmp3153, &tmp3157);
      if(tmp3158 > 0.5)
      {
        tmp3158 -= 1.0;
        tmp3157 += 1.0;
      }
      else if(tmp3158 < -0.5)
      {
        tmp3158 += 1.0;
        tmp3157 -= 1.0;
      }
      if(fabs(tmp3158) < 1e-10 && ((unsigned long)tmp3157 & 1))
      {
        tmp3154 = -pow(-tmp3152, tmp3155)*pow(tmp3152, tmp3156);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3152, tmp3153);
      }
    }
  }
  else
  {
    tmp3154 = pow(tmp3152, tmp3153);
  }
  if(isnan(tmp3154) || isinf(tmp3154))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3152, tmp3153);
  }tmp3159 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3154,"(r_init[185] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3159 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[185] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3159);
    }
  }
  (data->simulationInfo->realParameter[689] /* omega_c[185] PARAM */) = sqrt(tmp3159);
  TRACE_POP
}

/*
equation index: 13633
type: SIMPLE_ASSIGN
r_init[184] = r_min + 184.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13633};
  (data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (184.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13634
type: SIMPLE_ASSIGN
omega_c[184] = sqrt(G * Md / (r_init[184] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13634(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13634};
  modelica_real tmp3160;
  modelica_real tmp3161;
  modelica_real tmp3162;
  modelica_real tmp3163;
  modelica_real tmp3164;
  modelica_real tmp3165;
  modelica_real tmp3166;
  modelica_real tmp3167;
  modelica_real tmp3168;
  modelica_real tmp3169;
  tmp3160 = (data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */);
  tmp3161 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3162 = (tmp3160 * tmp3160) + (tmp3161 * tmp3161);
  tmp3163 = 1.5;
  if(tmp3162 < 0.0 && tmp3163 != 0.0)
  {
    tmp3165 = modf(tmp3163, &tmp3166);
    
    if(tmp3165 > 0.5)
    {
      tmp3165 -= 1.0;
      tmp3166 += 1.0;
    }
    else if(tmp3165 < -0.5)
    {
      tmp3165 += 1.0;
      tmp3166 -= 1.0;
    }
    
    if(fabs(tmp3165) < 1e-10)
      tmp3164 = pow(tmp3162, tmp3166);
    else
    {
      tmp3168 = modf(1.0/tmp3163, &tmp3167);
      if(tmp3168 > 0.5)
      {
        tmp3168 -= 1.0;
        tmp3167 += 1.0;
      }
      else if(tmp3168 < -0.5)
      {
        tmp3168 += 1.0;
        tmp3167 -= 1.0;
      }
      if(fabs(tmp3168) < 1e-10 && ((unsigned long)tmp3167 & 1))
      {
        tmp3164 = -pow(-tmp3162, tmp3165)*pow(tmp3162, tmp3166);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3162, tmp3163);
      }
    }
  }
  else
  {
    tmp3164 = pow(tmp3162, tmp3163);
  }
  if(isnan(tmp3164) || isinf(tmp3164))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3162, tmp3163);
  }tmp3169 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3164,"(r_init[184] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3169 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[184] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3169);
    }
  }
  (data->simulationInfo->realParameter[688] /* omega_c[184] PARAM */) = sqrt(tmp3169);
  TRACE_POP
}

/*
equation index: 13635
type: SIMPLE_ASSIGN
r_init[183] = r_min + 183.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13635};
  (data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (183.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13636
type: SIMPLE_ASSIGN
omega_c[183] = sqrt(G * Md / (r_init[183] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13636};
  modelica_real tmp3170;
  modelica_real tmp3171;
  modelica_real tmp3172;
  modelica_real tmp3173;
  modelica_real tmp3174;
  modelica_real tmp3175;
  modelica_real tmp3176;
  modelica_real tmp3177;
  modelica_real tmp3178;
  modelica_real tmp3179;
  tmp3170 = (data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */);
  tmp3171 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3172 = (tmp3170 * tmp3170) + (tmp3171 * tmp3171);
  tmp3173 = 1.5;
  if(tmp3172 < 0.0 && tmp3173 != 0.0)
  {
    tmp3175 = modf(tmp3173, &tmp3176);
    
    if(tmp3175 > 0.5)
    {
      tmp3175 -= 1.0;
      tmp3176 += 1.0;
    }
    else if(tmp3175 < -0.5)
    {
      tmp3175 += 1.0;
      tmp3176 -= 1.0;
    }
    
    if(fabs(tmp3175) < 1e-10)
      tmp3174 = pow(tmp3172, tmp3176);
    else
    {
      tmp3178 = modf(1.0/tmp3173, &tmp3177);
      if(tmp3178 > 0.5)
      {
        tmp3178 -= 1.0;
        tmp3177 += 1.0;
      }
      else if(tmp3178 < -0.5)
      {
        tmp3178 += 1.0;
        tmp3177 -= 1.0;
      }
      if(fabs(tmp3178) < 1e-10 && ((unsigned long)tmp3177 & 1))
      {
        tmp3174 = -pow(-tmp3172, tmp3175)*pow(tmp3172, tmp3176);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3172, tmp3173);
      }
    }
  }
  else
  {
    tmp3174 = pow(tmp3172, tmp3173);
  }
  if(isnan(tmp3174) || isinf(tmp3174))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3172, tmp3173);
  }tmp3179 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3174,"(r_init[183] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3179 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[183] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3179);
    }
  }
  (data->simulationInfo->realParameter[687] /* omega_c[183] PARAM */) = sqrt(tmp3179);
  TRACE_POP
}

/*
equation index: 13637
type: SIMPLE_ASSIGN
r_init[182] = r_min + 182.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13637};
  (data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (182.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13638
type: SIMPLE_ASSIGN
omega_c[182] = sqrt(G * Md / (r_init[182] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13638(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13638};
  modelica_real tmp3180;
  modelica_real tmp3181;
  modelica_real tmp3182;
  modelica_real tmp3183;
  modelica_real tmp3184;
  modelica_real tmp3185;
  modelica_real tmp3186;
  modelica_real tmp3187;
  modelica_real tmp3188;
  modelica_real tmp3189;
  tmp3180 = (data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */);
  tmp3181 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3182 = (tmp3180 * tmp3180) + (tmp3181 * tmp3181);
  tmp3183 = 1.5;
  if(tmp3182 < 0.0 && tmp3183 != 0.0)
  {
    tmp3185 = modf(tmp3183, &tmp3186);
    
    if(tmp3185 > 0.5)
    {
      tmp3185 -= 1.0;
      tmp3186 += 1.0;
    }
    else if(tmp3185 < -0.5)
    {
      tmp3185 += 1.0;
      tmp3186 -= 1.0;
    }
    
    if(fabs(tmp3185) < 1e-10)
      tmp3184 = pow(tmp3182, tmp3186);
    else
    {
      tmp3188 = modf(1.0/tmp3183, &tmp3187);
      if(tmp3188 > 0.5)
      {
        tmp3188 -= 1.0;
        tmp3187 += 1.0;
      }
      else if(tmp3188 < -0.5)
      {
        tmp3188 += 1.0;
        tmp3187 -= 1.0;
      }
      if(fabs(tmp3188) < 1e-10 && ((unsigned long)tmp3187 & 1))
      {
        tmp3184 = -pow(-tmp3182, tmp3185)*pow(tmp3182, tmp3186);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3182, tmp3183);
      }
    }
  }
  else
  {
    tmp3184 = pow(tmp3182, tmp3183);
  }
  if(isnan(tmp3184) || isinf(tmp3184))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3182, tmp3183);
  }tmp3189 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3184,"(r_init[182] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3189 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[182] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3189);
    }
  }
  (data->simulationInfo->realParameter[686] /* omega_c[182] PARAM */) = sqrt(tmp3189);
  TRACE_POP
}

/*
equation index: 13639
type: SIMPLE_ASSIGN
r_init[181] = r_min + 181.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13639};
  (data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (181.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13640
type: SIMPLE_ASSIGN
omega_c[181] = sqrt(G * Md / (r_init[181] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13640(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13640};
  modelica_real tmp3190;
  modelica_real tmp3191;
  modelica_real tmp3192;
  modelica_real tmp3193;
  modelica_real tmp3194;
  modelica_real tmp3195;
  modelica_real tmp3196;
  modelica_real tmp3197;
  modelica_real tmp3198;
  modelica_real tmp3199;
  tmp3190 = (data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */);
  tmp3191 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3192 = (tmp3190 * tmp3190) + (tmp3191 * tmp3191);
  tmp3193 = 1.5;
  if(tmp3192 < 0.0 && tmp3193 != 0.0)
  {
    tmp3195 = modf(tmp3193, &tmp3196);
    
    if(tmp3195 > 0.5)
    {
      tmp3195 -= 1.0;
      tmp3196 += 1.0;
    }
    else if(tmp3195 < -0.5)
    {
      tmp3195 += 1.0;
      tmp3196 -= 1.0;
    }
    
    if(fabs(tmp3195) < 1e-10)
      tmp3194 = pow(tmp3192, tmp3196);
    else
    {
      tmp3198 = modf(1.0/tmp3193, &tmp3197);
      if(tmp3198 > 0.5)
      {
        tmp3198 -= 1.0;
        tmp3197 += 1.0;
      }
      else if(tmp3198 < -0.5)
      {
        tmp3198 += 1.0;
        tmp3197 -= 1.0;
      }
      if(fabs(tmp3198) < 1e-10 && ((unsigned long)tmp3197 & 1))
      {
        tmp3194 = -pow(-tmp3192, tmp3195)*pow(tmp3192, tmp3196);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3192, tmp3193);
      }
    }
  }
  else
  {
    tmp3194 = pow(tmp3192, tmp3193);
  }
  if(isnan(tmp3194) || isinf(tmp3194))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3192, tmp3193);
  }tmp3199 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3194,"(r_init[181] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3199 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[181] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3199);
    }
  }
  (data->simulationInfo->realParameter[685] /* omega_c[181] PARAM */) = sqrt(tmp3199);
  TRACE_POP
}

/*
equation index: 13641
type: SIMPLE_ASSIGN
r_init[180] = r_min + 180.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13641};
  (data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (180.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13642
type: SIMPLE_ASSIGN
omega_c[180] = sqrt(G * Md / (r_init[180] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13642(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13642};
  modelica_real tmp3200;
  modelica_real tmp3201;
  modelica_real tmp3202;
  modelica_real tmp3203;
  modelica_real tmp3204;
  modelica_real tmp3205;
  modelica_real tmp3206;
  modelica_real tmp3207;
  modelica_real tmp3208;
  modelica_real tmp3209;
  tmp3200 = (data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */);
  tmp3201 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3202 = (tmp3200 * tmp3200) + (tmp3201 * tmp3201);
  tmp3203 = 1.5;
  if(tmp3202 < 0.0 && tmp3203 != 0.0)
  {
    tmp3205 = modf(tmp3203, &tmp3206);
    
    if(tmp3205 > 0.5)
    {
      tmp3205 -= 1.0;
      tmp3206 += 1.0;
    }
    else if(tmp3205 < -0.5)
    {
      tmp3205 += 1.0;
      tmp3206 -= 1.0;
    }
    
    if(fabs(tmp3205) < 1e-10)
      tmp3204 = pow(tmp3202, tmp3206);
    else
    {
      tmp3208 = modf(1.0/tmp3203, &tmp3207);
      if(tmp3208 > 0.5)
      {
        tmp3208 -= 1.0;
        tmp3207 += 1.0;
      }
      else if(tmp3208 < -0.5)
      {
        tmp3208 += 1.0;
        tmp3207 -= 1.0;
      }
      if(fabs(tmp3208) < 1e-10 && ((unsigned long)tmp3207 & 1))
      {
        tmp3204 = -pow(-tmp3202, tmp3205)*pow(tmp3202, tmp3206);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3202, tmp3203);
      }
    }
  }
  else
  {
    tmp3204 = pow(tmp3202, tmp3203);
  }
  if(isnan(tmp3204) || isinf(tmp3204))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3202, tmp3203);
  }tmp3209 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3204,"(r_init[180] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3209 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[180] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3209);
    }
  }
  (data->simulationInfo->realParameter[684] /* omega_c[180] PARAM */) = sqrt(tmp3209);
  TRACE_POP
}

/*
equation index: 13643
type: SIMPLE_ASSIGN
r_init[179] = r_min + 179.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13643};
  (data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (179.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13644
type: SIMPLE_ASSIGN
omega_c[179] = sqrt(G * Md / (r_init[179] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13644};
  modelica_real tmp3210;
  modelica_real tmp3211;
  modelica_real tmp3212;
  modelica_real tmp3213;
  modelica_real tmp3214;
  modelica_real tmp3215;
  modelica_real tmp3216;
  modelica_real tmp3217;
  modelica_real tmp3218;
  modelica_real tmp3219;
  tmp3210 = (data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */);
  tmp3211 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3212 = (tmp3210 * tmp3210) + (tmp3211 * tmp3211);
  tmp3213 = 1.5;
  if(tmp3212 < 0.0 && tmp3213 != 0.0)
  {
    tmp3215 = modf(tmp3213, &tmp3216);
    
    if(tmp3215 > 0.5)
    {
      tmp3215 -= 1.0;
      tmp3216 += 1.0;
    }
    else if(tmp3215 < -0.5)
    {
      tmp3215 += 1.0;
      tmp3216 -= 1.0;
    }
    
    if(fabs(tmp3215) < 1e-10)
      tmp3214 = pow(tmp3212, tmp3216);
    else
    {
      tmp3218 = modf(1.0/tmp3213, &tmp3217);
      if(tmp3218 > 0.5)
      {
        tmp3218 -= 1.0;
        tmp3217 += 1.0;
      }
      else if(tmp3218 < -0.5)
      {
        tmp3218 += 1.0;
        tmp3217 -= 1.0;
      }
      if(fabs(tmp3218) < 1e-10 && ((unsigned long)tmp3217 & 1))
      {
        tmp3214 = -pow(-tmp3212, tmp3215)*pow(tmp3212, tmp3216);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3212, tmp3213);
      }
    }
  }
  else
  {
    tmp3214 = pow(tmp3212, tmp3213);
  }
  if(isnan(tmp3214) || isinf(tmp3214))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3212, tmp3213);
  }tmp3219 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3214,"(r_init[179] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3219 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[179] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3219);
    }
  }
  (data->simulationInfo->realParameter[683] /* omega_c[179] PARAM */) = sqrt(tmp3219);
  TRACE_POP
}

/*
equation index: 13645
type: SIMPLE_ASSIGN
r_init[178] = r_min + 178.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13645};
  (data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (178.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13646
type: SIMPLE_ASSIGN
omega_c[178] = sqrt(G * Md / (r_init[178] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13646(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13646};
  modelica_real tmp3220;
  modelica_real tmp3221;
  modelica_real tmp3222;
  modelica_real tmp3223;
  modelica_real tmp3224;
  modelica_real tmp3225;
  modelica_real tmp3226;
  modelica_real tmp3227;
  modelica_real tmp3228;
  modelica_real tmp3229;
  tmp3220 = (data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */);
  tmp3221 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3222 = (tmp3220 * tmp3220) + (tmp3221 * tmp3221);
  tmp3223 = 1.5;
  if(tmp3222 < 0.0 && tmp3223 != 0.0)
  {
    tmp3225 = modf(tmp3223, &tmp3226);
    
    if(tmp3225 > 0.5)
    {
      tmp3225 -= 1.0;
      tmp3226 += 1.0;
    }
    else if(tmp3225 < -0.5)
    {
      tmp3225 += 1.0;
      tmp3226 -= 1.0;
    }
    
    if(fabs(tmp3225) < 1e-10)
      tmp3224 = pow(tmp3222, tmp3226);
    else
    {
      tmp3228 = modf(1.0/tmp3223, &tmp3227);
      if(tmp3228 > 0.5)
      {
        tmp3228 -= 1.0;
        tmp3227 += 1.0;
      }
      else if(tmp3228 < -0.5)
      {
        tmp3228 += 1.0;
        tmp3227 -= 1.0;
      }
      if(fabs(tmp3228) < 1e-10 && ((unsigned long)tmp3227 & 1))
      {
        tmp3224 = -pow(-tmp3222, tmp3225)*pow(tmp3222, tmp3226);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3222, tmp3223);
      }
    }
  }
  else
  {
    tmp3224 = pow(tmp3222, tmp3223);
  }
  if(isnan(tmp3224) || isinf(tmp3224))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3222, tmp3223);
  }tmp3229 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3224,"(r_init[178] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3229 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[178] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3229);
    }
  }
  (data->simulationInfo->realParameter[682] /* omega_c[178] PARAM */) = sqrt(tmp3229);
  TRACE_POP
}

/*
equation index: 13647
type: SIMPLE_ASSIGN
r_init[177] = r_min + 177.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13647};
  (data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (177.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13648
type: SIMPLE_ASSIGN
omega_c[177] = sqrt(G * Md / (r_init[177] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13648(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13648};
  modelica_real tmp3230;
  modelica_real tmp3231;
  modelica_real tmp3232;
  modelica_real tmp3233;
  modelica_real tmp3234;
  modelica_real tmp3235;
  modelica_real tmp3236;
  modelica_real tmp3237;
  modelica_real tmp3238;
  modelica_real tmp3239;
  tmp3230 = (data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */);
  tmp3231 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3232 = (tmp3230 * tmp3230) + (tmp3231 * tmp3231);
  tmp3233 = 1.5;
  if(tmp3232 < 0.0 && tmp3233 != 0.0)
  {
    tmp3235 = modf(tmp3233, &tmp3236);
    
    if(tmp3235 > 0.5)
    {
      tmp3235 -= 1.0;
      tmp3236 += 1.0;
    }
    else if(tmp3235 < -0.5)
    {
      tmp3235 += 1.0;
      tmp3236 -= 1.0;
    }
    
    if(fabs(tmp3235) < 1e-10)
      tmp3234 = pow(tmp3232, tmp3236);
    else
    {
      tmp3238 = modf(1.0/tmp3233, &tmp3237);
      if(tmp3238 > 0.5)
      {
        tmp3238 -= 1.0;
        tmp3237 += 1.0;
      }
      else if(tmp3238 < -0.5)
      {
        tmp3238 += 1.0;
        tmp3237 -= 1.0;
      }
      if(fabs(tmp3238) < 1e-10 && ((unsigned long)tmp3237 & 1))
      {
        tmp3234 = -pow(-tmp3232, tmp3235)*pow(tmp3232, tmp3236);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3232, tmp3233);
      }
    }
  }
  else
  {
    tmp3234 = pow(tmp3232, tmp3233);
  }
  if(isnan(tmp3234) || isinf(tmp3234))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3232, tmp3233);
  }tmp3239 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3234,"(r_init[177] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3239 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[177] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3239);
    }
  }
  (data->simulationInfo->realParameter[681] /* omega_c[177] PARAM */) = sqrt(tmp3239);
  TRACE_POP
}

/*
equation index: 13649
type: SIMPLE_ASSIGN
r_init[176] = r_min + 176.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13649};
  (data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (176.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13650
type: SIMPLE_ASSIGN
omega_c[176] = sqrt(G * Md / (r_init[176] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13650(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13650};
  modelica_real tmp3240;
  modelica_real tmp3241;
  modelica_real tmp3242;
  modelica_real tmp3243;
  modelica_real tmp3244;
  modelica_real tmp3245;
  modelica_real tmp3246;
  modelica_real tmp3247;
  modelica_real tmp3248;
  modelica_real tmp3249;
  tmp3240 = (data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */);
  tmp3241 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3242 = (tmp3240 * tmp3240) + (tmp3241 * tmp3241);
  tmp3243 = 1.5;
  if(tmp3242 < 0.0 && tmp3243 != 0.0)
  {
    tmp3245 = modf(tmp3243, &tmp3246);
    
    if(tmp3245 > 0.5)
    {
      tmp3245 -= 1.0;
      tmp3246 += 1.0;
    }
    else if(tmp3245 < -0.5)
    {
      tmp3245 += 1.0;
      tmp3246 -= 1.0;
    }
    
    if(fabs(tmp3245) < 1e-10)
      tmp3244 = pow(tmp3242, tmp3246);
    else
    {
      tmp3248 = modf(1.0/tmp3243, &tmp3247);
      if(tmp3248 > 0.5)
      {
        tmp3248 -= 1.0;
        tmp3247 += 1.0;
      }
      else if(tmp3248 < -0.5)
      {
        tmp3248 += 1.0;
        tmp3247 -= 1.0;
      }
      if(fabs(tmp3248) < 1e-10 && ((unsigned long)tmp3247 & 1))
      {
        tmp3244 = -pow(-tmp3242, tmp3245)*pow(tmp3242, tmp3246);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3242, tmp3243);
      }
    }
  }
  else
  {
    tmp3244 = pow(tmp3242, tmp3243);
  }
  if(isnan(tmp3244) || isinf(tmp3244))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3242, tmp3243);
  }tmp3249 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3244,"(r_init[176] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3249 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[176] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3249);
    }
  }
  (data->simulationInfo->realParameter[680] /* omega_c[176] PARAM */) = sqrt(tmp3249);
  TRACE_POP
}

/*
equation index: 13651
type: SIMPLE_ASSIGN
r_init[175] = r_min + 175.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13651};
  (data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (175.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13652
type: SIMPLE_ASSIGN
omega_c[175] = sqrt(G * Md / (r_init[175] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13652};
  modelica_real tmp3250;
  modelica_real tmp3251;
  modelica_real tmp3252;
  modelica_real tmp3253;
  modelica_real tmp3254;
  modelica_real tmp3255;
  modelica_real tmp3256;
  modelica_real tmp3257;
  modelica_real tmp3258;
  modelica_real tmp3259;
  tmp3250 = (data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */);
  tmp3251 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3252 = (tmp3250 * tmp3250) + (tmp3251 * tmp3251);
  tmp3253 = 1.5;
  if(tmp3252 < 0.0 && tmp3253 != 0.0)
  {
    tmp3255 = modf(tmp3253, &tmp3256);
    
    if(tmp3255 > 0.5)
    {
      tmp3255 -= 1.0;
      tmp3256 += 1.0;
    }
    else if(tmp3255 < -0.5)
    {
      tmp3255 += 1.0;
      tmp3256 -= 1.0;
    }
    
    if(fabs(tmp3255) < 1e-10)
      tmp3254 = pow(tmp3252, tmp3256);
    else
    {
      tmp3258 = modf(1.0/tmp3253, &tmp3257);
      if(tmp3258 > 0.5)
      {
        tmp3258 -= 1.0;
        tmp3257 += 1.0;
      }
      else if(tmp3258 < -0.5)
      {
        tmp3258 += 1.0;
        tmp3257 -= 1.0;
      }
      if(fabs(tmp3258) < 1e-10 && ((unsigned long)tmp3257 & 1))
      {
        tmp3254 = -pow(-tmp3252, tmp3255)*pow(tmp3252, tmp3256);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3252, tmp3253);
      }
    }
  }
  else
  {
    tmp3254 = pow(tmp3252, tmp3253);
  }
  if(isnan(tmp3254) || isinf(tmp3254))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3252, tmp3253);
  }tmp3259 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3254,"(r_init[175] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3259 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[175] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3259);
    }
  }
  (data->simulationInfo->realParameter[679] /* omega_c[175] PARAM */) = sqrt(tmp3259);
  TRACE_POP
}

/*
equation index: 13653
type: SIMPLE_ASSIGN
r_init[174] = r_min + 174.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13653};
  (data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (174.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13654
type: SIMPLE_ASSIGN
omega_c[174] = sqrt(G * Md / (r_init[174] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13654(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13654};
  modelica_real tmp3260;
  modelica_real tmp3261;
  modelica_real tmp3262;
  modelica_real tmp3263;
  modelica_real tmp3264;
  modelica_real tmp3265;
  modelica_real tmp3266;
  modelica_real tmp3267;
  modelica_real tmp3268;
  modelica_real tmp3269;
  tmp3260 = (data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */);
  tmp3261 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3262 = (tmp3260 * tmp3260) + (tmp3261 * tmp3261);
  tmp3263 = 1.5;
  if(tmp3262 < 0.0 && tmp3263 != 0.0)
  {
    tmp3265 = modf(tmp3263, &tmp3266);
    
    if(tmp3265 > 0.5)
    {
      tmp3265 -= 1.0;
      tmp3266 += 1.0;
    }
    else if(tmp3265 < -0.5)
    {
      tmp3265 += 1.0;
      tmp3266 -= 1.0;
    }
    
    if(fabs(tmp3265) < 1e-10)
      tmp3264 = pow(tmp3262, tmp3266);
    else
    {
      tmp3268 = modf(1.0/tmp3263, &tmp3267);
      if(tmp3268 > 0.5)
      {
        tmp3268 -= 1.0;
        tmp3267 += 1.0;
      }
      else if(tmp3268 < -0.5)
      {
        tmp3268 += 1.0;
        tmp3267 -= 1.0;
      }
      if(fabs(tmp3268) < 1e-10 && ((unsigned long)tmp3267 & 1))
      {
        tmp3264 = -pow(-tmp3262, tmp3265)*pow(tmp3262, tmp3266);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3262, tmp3263);
      }
    }
  }
  else
  {
    tmp3264 = pow(tmp3262, tmp3263);
  }
  if(isnan(tmp3264) || isinf(tmp3264))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3262, tmp3263);
  }tmp3269 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3264,"(r_init[174] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3269 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[174] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3269);
    }
  }
  (data->simulationInfo->realParameter[678] /* omega_c[174] PARAM */) = sqrt(tmp3269);
  TRACE_POP
}

/*
equation index: 13655
type: SIMPLE_ASSIGN
r_init[173] = r_min + 173.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13655};
  (data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (173.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13656
type: SIMPLE_ASSIGN
omega_c[173] = sqrt(G * Md / (r_init[173] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13656(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13656};
  modelica_real tmp3270;
  modelica_real tmp3271;
  modelica_real tmp3272;
  modelica_real tmp3273;
  modelica_real tmp3274;
  modelica_real tmp3275;
  modelica_real tmp3276;
  modelica_real tmp3277;
  modelica_real tmp3278;
  modelica_real tmp3279;
  tmp3270 = (data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */);
  tmp3271 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3272 = (tmp3270 * tmp3270) + (tmp3271 * tmp3271);
  tmp3273 = 1.5;
  if(tmp3272 < 0.0 && tmp3273 != 0.0)
  {
    tmp3275 = modf(tmp3273, &tmp3276);
    
    if(tmp3275 > 0.5)
    {
      tmp3275 -= 1.0;
      tmp3276 += 1.0;
    }
    else if(tmp3275 < -0.5)
    {
      tmp3275 += 1.0;
      tmp3276 -= 1.0;
    }
    
    if(fabs(tmp3275) < 1e-10)
      tmp3274 = pow(tmp3272, tmp3276);
    else
    {
      tmp3278 = modf(1.0/tmp3273, &tmp3277);
      if(tmp3278 > 0.5)
      {
        tmp3278 -= 1.0;
        tmp3277 += 1.0;
      }
      else if(tmp3278 < -0.5)
      {
        tmp3278 += 1.0;
        tmp3277 -= 1.0;
      }
      if(fabs(tmp3278) < 1e-10 && ((unsigned long)tmp3277 & 1))
      {
        tmp3274 = -pow(-tmp3272, tmp3275)*pow(tmp3272, tmp3276);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3272, tmp3273);
      }
    }
  }
  else
  {
    tmp3274 = pow(tmp3272, tmp3273);
  }
  if(isnan(tmp3274) || isinf(tmp3274))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3272, tmp3273);
  }tmp3279 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3274,"(r_init[173] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3279 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[173] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3279);
    }
  }
  (data->simulationInfo->realParameter[677] /* omega_c[173] PARAM */) = sqrt(tmp3279);
  TRACE_POP
}

/*
equation index: 13657
type: SIMPLE_ASSIGN
r_init[172] = r_min + 172.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13657};
  (data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (172.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13658
type: SIMPLE_ASSIGN
omega_c[172] = sqrt(G * Md / (r_init[172] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13658(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13658};
  modelica_real tmp3280;
  modelica_real tmp3281;
  modelica_real tmp3282;
  modelica_real tmp3283;
  modelica_real tmp3284;
  modelica_real tmp3285;
  modelica_real tmp3286;
  modelica_real tmp3287;
  modelica_real tmp3288;
  modelica_real tmp3289;
  tmp3280 = (data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */);
  tmp3281 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3282 = (tmp3280 * tmp3280) + (tmp3281 * tmp3281);
  tmp3283 = 1.5;
  if(tmp3282 < 0.0 && tmp3283 != 0.0)
  {
    tmp3285 = modf(tmp3283, &tmp3286);
    
    if(tmp3285 > 0.5)
    {
      tmp3285 -= 1.0;
      tmp3286 += 1.0;
    }
    else if(tmp3285 < -0.5)
    {
      tmp3285 += 1.0;
      tmp3286 -= 1.0;
    }
    
    if(fabs(tmp3285) < 1e-10)
      tmp3284 = pow(tmp3282, tmp3286);
    else
    {
      tmp3288 = modf(1.0/tmp3283, &tmp3287);
      if(tmp3288 > 0.5)
      {
        tmp3288 -= 1.0;
        tmp3287 += 1.0;
      }
      else if(tmp3288 < -0.5)
      {
        tmp3288 += 1.0;
        tmp3287 -= 1.0;
      }
      if(fabs(tmp3288) < 1e-10 && ((unsigned long)tmp3287 & 1))
      {
        tmp3284 = -pow(-tmp3282, tmp3285)*pow(tmp3282, tmp3286);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3282, tmp3283);
      }
    }
  }
  else
  {
    tmp3284 = pow(tmp3282, tmp3283);
  }
  if(isnan(tmp3284) || isinf(tmp3284))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3282, tmp3283);
  }tmp3289 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3284,"(r_init[172] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3289 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[172] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3289);
    }
  }
  (data->simulationInfo->realParameter[676] /* omega_c[172] PARAM */) = sqrt(tmp3289);
  TRACE_POP
}

/*
equation index: 13659
type: SIMPLE_ASSIGN
r_init[171] = r_min + 171.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13659};
  (data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (171.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13660
type: SIMPLE_ASSIGN
omega_c[171] = sqrt(G * Md / (r_init[171] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13660};
  modelica_real tmp3290;
  modelica_real tmp3291;
  modelica_real tmp3292;
  modelica_real tmp3293;
  modelica_real tmp3294;
  modelica_real tmp3295;
  modelica_real tmp3296;
  modelica_real tmp3297;
  modelica_real tmp3298;
  modelica_real tmp3299;
  tmp3290 = (data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */);
  tmp3291 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3292 = (tmp3290 * tmp3290) + (tmp3291 * tmp3291);
  tmp3293 = 1.5;
  if(tmp3292 < 0.0 && tmp3293 != 0.0)
  {
    tmp3295 = modf(tmp3293, &tmp3296);
    
    if(tmp3295 > 0.5)
    {
      tmp3295 -= 1.0;
      tmp3296 += 1.0;
    }
    else if(tmp3295 < -0.5)
    {
      tmp3295 += 1.0;
      tmp3296 -= 1.0;
    }
    
    if(fabs(tmp3295) < 1e-10)
      tmp3294 = pow(tmp3292, tmp3296);
    else
    {
      tmp3298 = modf(1.0/tmp3293, &tmp3297);
      if(tmp3298 > 0.5)
      {
        tmp3298 -= 1.0;
        tmp3297 += 1.0;
      }
      else if(tmp3298 < -0.5)
      {
        tmp3298 += 1.0;
        tmp3297 -= 1.0;
      }
      if(fabs(tmp3298) < 1e-10 && ((unsigned long)tmp3297 & 1))
      {
        tmp3294 = -pow(-tmp3292, tmp3295)*pow(tmp3292, tmp3296);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3292, tmp3293);
      }
    }
  }
  else
  {
    tmp3294 = pow(tmp3292, tmp3293);
  }
  if(isnan(tmp3294) || isinf(tmp3294))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3292, tmp3293);
  }tmp3299 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3294,"(r_init[171] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3299 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[171] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3299);
    }
  }
  (data->simulationInfo->realParameter[675] /* omega_c[171] PARAM */) = sqrt(tmp3299);
  TRACE_POP
}

/*
equation index: 13661
type: SIMPLE_ASSIGN
r_init[170] = r_min + 170.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13661};
  (data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (170.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13662
type: SIMPLE_ASSIGN
omega_c[170] = sqrt(G * Md / (r_init[170] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13662(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13662};
  modelica_real tmp3300;
  modelica_real tmp3301;
  modelica_real tmp3302;
  modelica_real tmp3303;
  modelica_real tmp3304;
  modelica_real tmp3305;
  modelica_real tmp3306;
  modelica_real tmp3307;
  modelica_real tmp3308;
  modelica_real tmp3309;
  tmp3300 = (data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */);
  tmp3301 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3302 = (tmp3300 * tmp3300) + (tmp3301 * tmp3301);
  tmp3303 = 1.5;
  if(tmp3302 < 0.0 && tmp3303 != 0.0)
  {
    tmp3305 = modf(tmp3303, &tmp3306);
    
    if(tmp3305 > 0.5)
    {
      tmp3305 -= 1.0;
      tmp3306 += 1.0;
    }
    else if(tmp3305 < -0.5)
    {
      tmp3305 += 1.0;
      tmp3306 -= 1.0;
    }
    
    if(fabs(tmp3305) < 1e-10)
      tmp3304 = pow(tmp3302, tmp3306);
    else
    {
      tmp3308 = modf(1.0/tmp3303, &tmp3307);
      if(tmp3308 > 0.5)
      {
        tmp3308 -= 1.0;
        tmp3307 += 1.0;
      }
      else if(tmp3308 < -0.5)
      {
        tmp3308 += 1.0;
        tmp3307 -= 1.0;
      }
      if(fabs(tmp3308) < 1e-10 && ((unsigned long)tmp3307 & 1))
      {
        tmp3304 = -pow(-tmp3302, tmp3305)*pow(tmp3302, tmp3306);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3302, tmp3303);
      }
    }
  }
  else
  {
    tmp3304 = pow(tmp3302, tmp3303);
  }
  if(isnan(tmp3304) || isinf(tmp3304))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3302, tmp3303);
  }tmp3309 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3304,"(r_init[170] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3309 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[170] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3309);
    }
  }
  (data->simulationInfo->realParameter[674] /* omega_c[170] PARAM */) = sqrt(tmp3309);
  TRACE_POP
}

/*
equation index: 13663
type: SIMPLE_ASSIGN
r_init[169] = r_min + 169.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13663};
  (data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (169.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13664
type: SIMPLE_ASSIGN
omega_c[169] = sqrt(G * Md / (r_init[169] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13664(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13664};
  modelica_real tmp3310;
  modelica_real tmp3311;
  modelica_real tmp3312;
  modelica_real tmp3313;
  modelica_real tmp3314;
  modelica_real tmp3315;
  modelica_real tmp3316;
  modelica_real tmp3317;
  modelica_real tmp3318;
  modelica_real tmp3319;
  tmp3310 = (data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */);
  tmp3311 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3312 = (tmp3310 * tmp3310) + (tmp3311 * tmp3311);
  tmp3313 = 1.5;
  if(tmp3312 < 0.0 && tmp3313 != 0.0)
  {
    tmp3315 = modf(tmp3313, &tmp3316);
    
    if(tmp3315 > 0.5)
    {
      tmp3315 -= 1.0;
      tmp3316 += 1.0;
    }
    else if(tmp3315 < -0.5)
    {
      tmp3315 += 1.0;
      tmp3316 -= 1.0;
    }
    
    if(fabs(tmp3315) < 1e-10)
      tmp3314 = pow(tmp3312, tmp3316);
    else
    {
      tmp3318 = modf(1.0/tmp3313, &tmp3317);
      if(tmp3318 > 0.5)
      {
        tmp3318 -= 1.0;
        tmp3317 += 1.0;
      }
      else if(tmp3318 < -0.5)
      {
        tmp3318 += 1.0;
        tmp3317 -= 1.0;
      }
      if(fabs(tmp3318) < 1e-10 && ((unsigned long)tmp3317 & 1))
      {
        tmp3314 = -pow(-tmp3312, tmp3315)*pow(tmp3312, tmp3316);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3312, tmp3313);
      }
    }
  }
  else
  {
    tmp3314 = pow(tmp3312, tmp3313);
  }
  if(isnan(tmp3314) || isinf(tmp3314))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3312, tmp3313);
  }tmp3319 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3314,"(r_init[169] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3319 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[169] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3319);
    }
  }
  (data->simulationInfo->realParameter[673] /* omega_c[169] PARAM */) = sqrt(tmp3319);
  TRACE_POP
}

/*
equation index: 13665
type: SIMPLE_ASSIGN
r_init[168] = r_min + 168.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13665};
  (data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (168.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13666
type: SIMPLE_ASSIGN
omega_c[168] = sqrt(G * Md / (r_init[168] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13666(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13666};
  modelica_real tmp3320;
  modelica_real tmp3321;
  modelica_real tmp3322;
  modelica_real tmp3323;
  modelica_real tmp3324;
  modelica_real tmp3325;
  modelica_real tmp3326;
  modelica_real tmp3327;
  modelica_real tmp3328;
  modelica_real tmp3329;
  tmp3320 = (data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */);
  tmp3321 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3322 = (tmp3320 * tmp3320) + (tmp3321 * tmp3321);
  tmp3323 = 1.5;
  if(tmp3322 < 0.0 && tmp3323 != 0.0)
  {
    tmp3325 = modf(tmp3323, &tmp3326);
    
    if(tmp3325 > 0.5)
    {
      tmp3325 -= 1.0;
      tmp3326 += 1.0;
    }
    else if(tmp3325 < -0.5)
    {
      tmp3325 += 1.0;
      tmp3326 -= 1.0;
    }
    
    if(fabs(tmp3325) < 1e-10)
      tmp3324 = pow(tmp3322, tmp3326);
    else
    {
      tmp3328 = modf(1.0/tmp3323, &tmp3327);
      if(tmp3328 > 0.5)
      {
        tmp3328 -= 1.0;
        tmp3327 += 1.0;
      }
      else if(tmp3328 < -0.5)
      {
        tmp3328 += 1.0;
        tmp3327 -= 1.0;
      }
      if(fabs(tmp3328) < 1e-10 && ((unsigned long)tmp3327 & 1))
      {
        tmp3324 = -pow(-tmp3322, tmp3325)*pow(tmp3322, tmp3326);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3322, tmp3323);
      }
    }
  }
  else
  {
    tmp3324 = pow(tmp3322, tmp3323);
  }
  if(isnan(tmp3324) || isinf(tmp3324))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3322, tmp3323);
  }tmp3329 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3324,"(r_init[168] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3329 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[168] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3329);
    }
  }
  (data->simulationInfo->realParameter[672] /* omega_c[168] PARAM */) = sqrt(tmp3329);
  TRACE_POP
}

/*
equation index: 13667
type: SIMPLE_ASSIGN
r_init[167] = r_min + 167.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13667};
  (data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (167.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13668
type: SIMPLE_ASSIGN
omega_c[167] = sqrt(G * Md / (r_init[167] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13668};
  modelica_real tmp3330;
  modelica_real tmp3331;
  modelica_real tmp3332;
  modelica_real tmp3333;
  modelica_real tmp3334;
  modelica_real tmp3335;
  modelica_real tmp3336;
  modelica_real tmp3337;
  modelica_real tmp3338;
  modelica_real tmp3339;
  tmp3330 = (data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */);
  tmp3331 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3332 = (tmp3330 * tmp3330) + (tmp3331 * tmp3331);
  tmp3333 = 1.5;
  if(tmp3332 < 0.0 && tmp3333 != 0.0)
  {
    tmp3335 = modf(tmp3333, &tmp3336);
    
    if(tmp3335 > 0.5)
    {
      tmp3335 -= 1.0;
      tmp3336 += 1.0;
    }
    else if(tmp3335 < -0.5)
    {
      tmp3335 += 1.0;
      tmp3336 -= 1.0;
    }
    
    if(fabs(tmp3335) < 1e-10)
      tmp3334 = pow(tmp3332, tmp3336);
    else
    {
      tmp3338 = modf(1.0/tmp3333, &tmp3337);
      if(tmp3338 > 0.5)
      {
        tmp3338 -= 1.0;
        tmp3337 += 1.0;
      }
      else if(tmp3338 < -0.5)
      {
        tmp3338 += 1.0;
        tmp3337 -= 1.0;
      }
      if(fabs(tmp3338) < 1e-10 && ((unsigned long)tmp3337 & 1))
      {
        tmp3334 = -pow(-tmp3332, tmp3335)*pow(tmp3332, tmp3336);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3332, tmp3333);
      }
    }
  }
  else
  {
    tmp3334 = pow(tmp3332, tmp3333);
  }
  if(isnan(tmp3334) || isinf(tmp3334))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3332, tmp3333);
  }tmp3339 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3334,"(r_init[167] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3339 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[167] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3339);
    }
  }
  (data->simulationInfo->realParameter[671] /* omega_c[167] PARAM */) = sqrt(tmp3339);
  TRACE_POP
}

/*
equation index: 13669
type: SIMPLE_ASSIGN
r_init[166] = r_min + 166.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13669};
  (data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (166.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13670
type: SIMPLE_ASSIGN
omega_c[166] = sqrt(G * Md / (r_init[166] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13670(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13670};
  modelica_real tmp3340;
  modelica_real tmp3341;
  modelica_real tmp3342;
  modelica_real tmp3343;
  modelica_real tmp3344;
  modelica_real tmp3345;
  modelica_real tmp3346;
  modelica_real tmp3347;
  modelica_real tmp3348;
  modelica_real tmp3349;
  tmp3340 = (data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */);
  tmp3341 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3342 = (tmp3340 * tmp3340) + (tmp3341 * tmp3341);
  tmp3343 = 1.5;
  if(tmp3342 < 0.0 && tmp3343 != 0.0)
  {
    tmp3345 = modf(tmp3343, &tmp3346);
    
    if(tmp3345 > 0.5)
    {
      tmp3345 -= 1.0;
      tmp3346 += 1.0;
    }
    else if(tmp3345 < -0.5)
    {
      tmp3345 += 1.0;
      tmp3346 -= 1.0;
    }
    
    if(fabs(tmp3345) < 1e-10)
      tmp3344 = pow(tmp3342, tmp3346);
    else
    {
      tmp3348 = modf(1.0/tmp3343, &tmp3347);
      if(tmp3348 > 0.5)
      {
        tmp3348 -= 1.0;
        tmp3347 += 1.0;
      }
      else if(tmp3348 < -0.5)
      {
        tmp3348 += 1.0;
        tmp3347 -= 1.0;
      }
      if(fabs(tmp3348) < 1e-10 && ((unsigned long)tmp3347 & 1))
      {
        tmp3344 = -pow(-tmp3342, tmp3345)*pow(tmp3342, tmp3346);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3342, tmp3343);
      }
    }
  }
  else
  {
    tmp3344 = pow(tmp3342, tmp3343);
  }
  if(isnan(tmp3344) || isinf(tmp3344))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3342, tmp3343);
  }tmp3349 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3344,"(r_init[166] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3349 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[166] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3349);
    }
  }
  (data->simulationInfo->realParameter[670] /* omega_c[166] PARAM */) = sqrt(tmp3349);
  TRACE_POP
}

/*
equation index: 13671
type: SIMPLE_ASSIGN
r_init[165] = r_min + 165.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13671};
  (data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (165.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13672
type: SIMPLE_ASSIGN
omega_c[165] = sqrt(G * Md / (r_init[165] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13672(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13672};
  modelica_real tmp3350;
  modelica_real tmp3351;
  modelica_real tmp3352;
  modelica_real tmp3353;
  modelica_real tmp3354;
  modelica_real tmp3355;
  modelica_real tmp3356;
  modelica_real tmp3357;
  modelica_real tmp3358;
  modelica_real tmp3359;
  tmp3350 = (data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */);
  tmp3351 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3352 = (tmp3350 * tmp3350) + (tmp3351 * tmp3351);
  tmp3353 = 1.5;
  if(tmp3352 < 0.0 && tmp3353 != 0.0)
  {
    tmp3355 = modf(tmp3353, &tmp3356);
    
    if(tmp3355 > 0.5)
    {
      tmp3355 -= 1.0;
      tmp3356 += 1.0;
    }
    else if(tmp3355 < -0.5)
    {
      tmp3355 += 1.0;
      tmp3356 -= 1.0;
    }
    
    if(fabs(tmp3355) < 1e-10)
      tmp3354 = pow(tmp3352, tmp3356);
    else
    {
      tmp3358 = modf(1.0/tmp3353, &tmp3357);
      if(tmp3358 > 0.5)
      {
        tmp3358 -= 1.0;
        tmp3357 += 1.0;
      }
      else if(tmp3358 < -0.5)
      {
        tmp3358 += 1.0;
        tmp3357 -= 1.0;
      }
      if(fabs(tmp3358) < 1e-10 && ((unsigned long)tmp3357 & 1))
      {
        tmp3354 = -pow(-tmp3352, tmp3355)*pow(tmp3352, tmp3356);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3352, tmp3353);
      }
    }
  }
  else
  {
    tmp3354 = pow(tmp3352, tmp3353);
  }
  if(isnan(tmp3354) || isinf(tmp3354))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3352, tmp3353);
  }tmp3359 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3354,"(r_init[165] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3359 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[165] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3359);
    }
  }
  (data->simulationInfo->realParameter[669] /* omega_c[165] PARAM */) = sqrt(tmp3359);
  TRACE_POP
}

/*
equation index: 13673
type: SIMPLE_ASSIGN
r_init[164] = r_min + 164.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13673};
  (data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (164.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13674
type: SIMPLE_ASSIGN
omega_c[164] = sqrt(G * Md / (r_init[164] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13674(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13674};
  modelica_real tmp3360;
  modelica_real tmp3361;
  modelica_real tmp3362;
  modelica_real tmp3363;
  modelica_real tmp3364;
  modelica_real tmp3365;
  modelica_real tmp3366;
  modelica_real tmp3367;
  modelica_real tmp3368;
  modelica_real tmp3369;
  tmp3360 = (data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */);
  tmp3361 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3362 = (tmp3360 * tmp3360) + (tmp3361 * tmp3361);
  tmp3363 = 1.5;
  if(tmp3362 < 0.0 && tmp3363 != 0.0)
  {
    tmp3365 = modf(tmp3363, &tmp3366);
    
    if(tmp3365 > 0.5)
    {
      tmp3365 -= 1.0;
      tmp3366 += 1.0;
    }
    else if(tmp3365 < -0.5)
    {
      tmp3365 += 1.0;
      tmp3366 -= 1.0;
    }
    
    if(fabs(tmp3365) < 1e-10)
      tmp3364 = pow(tmp3362, tmp3366);
    else
    {
      tmp3368 = modf(1.0/tmp3363, &tmp3367);
      if(tmp3368 > 0.5)
      {
        tmp3368 -= 1.0;
        tmp3367 += 1.0;
      }
      else if(tmp3368 < -0.5)
      {
        tmp3368 += 1.0;
        tmp3367 -= 1.0;
      }
      if(fabs(tmp3368) < 1e-10 && ((unsigned long)tmp3367 & 1))
      {
        tmp3364 = -pow(-tmp3362, tmp3365)*pow(tmp3362, tmp3366);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3362, tmp3363);
      }
    }
  }
  else
  {
    tmp3364 = pow(tmp3362, tmp3363);
  }
  if(isnan(tmp3364) || isinf(tmp3364))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3362, tmp3363);
  }tmp3369 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3364,"(r_init[164] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3369 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[164] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3369);
    }
  }
  (data->simulationInfo->realParameter[668] /* omega_c[164] PARAM */) = sqrt(tmp3369);
  TRACE_POP
}

/*
equation index: 13675
type: SIMPLE_ASSIGN
r_init[163] = r_min + 163.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13675};
  (data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (163.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13676
type: SIMPLE_ASSIGN
omega_c[163] = sqrt(G * Md / (r_init[163] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13676};
  modelica_real tmp3370;
  modelica_real tmp3371;
  modelica_real tmp3372;
  modelica_real tmp3373;
  modelica_real tmp3374;
  modelica_real tmp3375;
  modelica_real tmp3376;
  modelica_real tmp3377;
  modelica_real tmp3378;
  modelica_real tmp3379;
  tmp3370 = (data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */);
  tmp3371 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3372 = (tmp3370 * tmp3370) + (tmp3371 * tmp3371);
  tmp3373 = 1.5;
  if(tmp3372 < 0.0 && tmp3373 != 0.0)
  {
    tmp3375 = modf(tmp3373, &tmp3376);
    
    if(tmp3375 > 0.5)
    {
      tmp3375 -= 1.0;
      tmp3376 += 1.0;
    }
    else if(tmp3375 < -0.5)
    {
      tmp3375 += 1.0;
      tmp3376 -= 1.0;
    }
    
    if(fabs(tmp3375) < 1e-10)
      tmp3374 = pow(tmp3372, tmp3376);
    else
    {
      tmp3378 = modf(1.0/tmp3373, &tmp3377);
      if(tmp3378 > 0.5)
      {
        tmp3378 -= 1.0;
        tmp3377 += 1.0;
      }
      else if(tmp3378 < -0.5)
      {
        tmp3378 += 1.0;
        tmp3377 -= 1.0;
      }
      if(fabs(tmp3378) < 1e-10 && ((unsigned long)tmp3377 & 1))
      {
        tmp3374 = -pow(-tmp3372, tmp3375)*pow(tmp3372, tmp3376);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3372, tmp3373);
      }
    }
  }
  else
  {
    tmp3374 = pow(tmp3372, tmp3373);
  }
  if(isnan(tmp3374) || isinf(tmp3374))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3372, tmp3373);
  }tmp3379 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3374,"(r_init[163] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3379 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[163] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3379);
    }
  }
  (data->simulationInfo->realParameter[667] /* omega_c[163] PARAM */) = sqrt(tmp3379);
  TRACE_POP
}

/*
equation index: 13677
type: SIMPLE_ASSIGN
r_init[162] = r_min + 162.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13677};
  (data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (162.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13678
type: SIMPLE_ASSIGN
omega_c[162] = sqrt(G * Md / (r_init[162] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13678(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13678};
  modelica_real tmp3380;
  modelica_real tmp3381;
  modelica_real tmp3382;
  modelica_real tmp3383;
  modelica_real tmp3384;
  modelica_real tmp3385;
  modelica_real tmp3386;
  modelica_real tmp3387;
  modelica_real tmp3388;
  modelica_real tmp3389;
  tmp3380 = (data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */);
  tmp3381 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3382 = (tmp3380 * tmp3380) + (tmp3381 * tmp3381);
  tmp3383 = 1.5;
  if(tmp3382 < 0.0 && tmp3383 != 0.0)
  {
    tmp3385 = modf(tmp3383, &tmp3386);
    
    if(tmp3385 > 0.5)
    {
      tmp3385 -= 1.0;
      tmp3386 += 1.0;
    }
    else if(tmp3385 < -0.5)
    {
      tmp3385 += 1.0;
      tmp3386 -= 1.0;
    }
    
    if(fabs(tmp3385) < 1e-10)
      tmp3384 = pow(tmp3382, tmp3386);
    else
    {
      tmp3388 = modf(1.0/tmp3383, &tmp3387);
      if(tmp3388 > 0.5)
      {
        tmp3388 -= 1.0;
        tmp3387 += 1.0;
      }
      else if(tmp3388 < -0.5)
      {
        tmp3388 += 1.0;
        tmp3387 -= 1.0;
      }
      if(fabs(tmp3388) < 1e-10 && ((unsigned long)tmp3387 & 1))
      {
        tmp3384 = -pow(-tmp3382, tmp3385)*pow(tmp3382, tmp3386);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3382, tmp3383);
      }
    }
  }
  else
  {
    tmp3384 = pow(tmp3382, tmp3383);
  }
  if(isnan(tmp3384) || isinf(tmp3384))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3382, tmp3383);
  }tmp3389 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3384,"(r_init[162] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3389 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[162] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3389);
    }
  }
  (data->simulationInfo->realParameter[666] /* omega_c[162] PARAM */) = sqrt(tmp3389);
  TRACE_POP
}

/*
equation index: 13679
type: SIMPLE_ASSIGN
r_init[161] = r_min + 161.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13679};
  (data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (161.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13680
type: SIMPLE_ASSIGN
omega_c[161] = sqrt(G * Md / (r_init[161] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13680(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13680};
  modelica_real tmp3390;
  modelica_real tmp3391;
  modelica_real tmp3392;
  modelica_real tmp3393;
  modelica_real tmp3394;
  modelica_real tmp3395;
  modelica_real tmp3396;
  modelica_real tmp3397;
  modelica_real tmp3398;
  modelica_real tmp3399;
  tmp3390 = (data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */);
  tmp3391 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3392 = (tmp3390 * tmp3390) + (tmp3391 * tmp3391);
  tmp3393 = 1.5;
  if(tmp3392 < 0.0 && tmp3393 != 0.0)
  {
    tmp3395 = modf(tmp3393, &tmp3396);
    
    if(tmp3395 > 0.5)
    {
      tmp3395 -= 1.0;
      tmp3396 += 1.0;
    }
    else if(tmp3395 < -0.5)
    {
      tmp3395 += 1.0;
      tmp3396 -= 1.0;
    }
    
    if(fabs(tmp3395) < 1e-10)
      tmp3394 = pow(tmp3392, tmp3396);
    else
    {
      tmp3398 = modf(1.0/tmp3393, &tmp3397);
      if(tmp3398 > 0.5)
      {
        tmp3398 -= 1.0;
        tmp3397 += 1.0;
      }
      else if(tmp3398 < -0.5)
      {
        tmp3398 += 1.0;
        tmp3397 -= 1.0;
      }
      if(fabs(tmp3398) < 1e-10 && ((unsigned long)tmp3397 & 1))
      {
        tmp3394 = -pow(-tmp3392, tmp3395)*pow(tmp3392, tmp3396);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3392, tmp3393);
      }
    }
  }
  else
  {
    tmp3394 = pow(tmp3392, tmp3393);
  }
  if(isnan(tmp3394) || isinf(tmp3394))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3392, tmp3393);
  }tmp3399 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3394,"(r_init[161] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3399 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[161] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3399);
    }
  }
  (data->simulationInfo->realParameter[665] /* omega_c[161] PARAM */) = sqrt(tmp3399);
  TRACE_POP
}

/*
equation index: 13681
type: SIMPLE_ASSIGN
r_init[160] = r_min + 160.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13681};
  (data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (160.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13682
type: SIMPLE_ASSIGN
omega_c[160] = sqrt(G * Md / (r_init[160] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13682(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13682};
  modelica_real tmp3400;
  modelica_real tmp3401;
  modelica_real tmp3402;
  modelica_real tmp3403;
  modelica_real tmp3404;
  modelica_real tmp3405;
  modelica_real tmp3406;
  modelica_real tmp3407;
  modelica_real tmp3408;
  modelica_real tmp3409;
  tmp3400 = (data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */);
  tmp3401 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3402 = (tmp3400 * tmp3400) + (tmp3401 * tmp3401);
  tmp3403 = 1.5;
  if(tmp3402 < 0.0 && tmp3403 != 0.0)
  {
    tmp3405 = modf(tmp3403, &tmp3406);
    
    if(tmp3405 > 0.5)
    {
      tmp3405 -= 1.0;
      tmp3406 += 1.0;
    }
    else if(tmp3405 < -0.5)
    {
      tmp3405 += 1.0;
      tmp3406 -= 1.0;
    }
    
    if(fabs(tmp3405) < 1e-10)
      tmp3404 = pow(tmp3402, tmp3406);
    else
    {
      tmp3408 = modf(1.0/tmp3403, &tmp3407);
      if(tmp3408 > 0.5)
      {
        tmp3408 -= 1.0;
        tmp3407 += 1.0;
      }
      else if(tmp3408 < -0.5)
      {
        tmp3408 += 1.0;
        tmp3407 -= 1.0;
      }
      if(fabs(tmp3408) < 1e-10 && ((unsigned long)tmp3407 & 1))
      {
        tmp3404 = -pow(-tmp3402, tmp3405)*pow(tmp3402, tmp3406);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3402, tmp3403);
      }
    }
  }
  else
  {
    tmp3404 = pow(tmp3402, tmp3403);
  }
  if(isnan(tmp3404) || isinf(tmp3404))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3402, tmp3403);
  }tmp3409 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3404,"(r_init[160] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3409 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[160] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3409);
    }
  }
  (data->simulationInfo->realParameter[664] /* omega_c[160] PARAM */) = sqrt(tmp3409);
  TRACE_POP
}

/*
equation index: 13683
type: SIMPLE_ASSIGN
r_init[159] = r_min + 159.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13683};
  (data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (159.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13684
type: SIMPLE_ASSIGN
omega_c[159] = sqrt(G * Md / (r_init[159] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13684};
  modelica_real tmp3410;
  modelica_real tmp3411;
  modelica_real tmp3412;
  modelica_real tmp3413;
  modelica_real tmp3414;
  modelica_real tmp3415;
  modelica_real tmp3416;
  modelica_real tmp3417;
  modelica_real tmp3418;
  modelica_real tmp3419;
  tmp3410 = (data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */);
  tmp3411 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3412 = (tmp3410 * tmp3410) + (tmp3411 * tmp3411);
  tmp3413 = 1.5;
  if(tmp3412 < 0.0 && tmp3413 != 0.0)
  {
    tmp3415 = modf(tmp3413, &tmp3416);
    
    if(tmp3415 > 0.5)
    {
      tmp3415 -= 1.0;
      tmp3416 += 1.0;
    }
    else if(tmp3415 < -0.5)
    {
      tmp3415 += 1.0;
      tmp3416 -= 1.0;
    }
    
    if(fabs(tmp3415) < 1e-10)
      tmp3414 = pow(tmp3412, tmp3416);
    else
    {
      tmp3418 = modf(1.0/tmp3413, &tmp3417);
      if(tmp3418 > 0.5)
      {
        tmp3418 -= 1.0;
        tmp3417 += 1.0;
      }
      else if(tmp3418 < -0.5)
      {
        tmp3418 += 1.0;
        tmp3417 -= 1.0;
      }
      if(fabs(tmp3418) < 1e-10 && ((unsigned long)tmp3417 & 1))
      {
        tmp3414 = -pow(-tmp3412, tmp3415)*pow(tmp3412, tmp3416);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3412, tmp3413);
      }
    }
  }
  else
  {
    tmp3414 = pow(tmp3412, tmp3413);
  }
  if(isnan(tmp3414) || isinf(tmp3414))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3412, tmp3413);
  }tmp3419 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3414,"(r_init[159] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3419 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[159] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3419);
    }
  }
  (data->simulationInfo->realParameter[663] /* omega_c[159] PARAM */) = sqrt(tmp3419);
  TRACE_POP
}

/*
equation index: 13685
type: SIMPLE_ASSIGN
r_init[158] = r_min + 158.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13685};
  (data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (158.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13686
type: SIMPLE_ASSIGN
omega_c[158] = sqrt(G * Md / (r_init[158] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13686(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13686};
  modelica_real tmp3420;
  modelica_real tmp3421;
  modelica_real tmp3422;
  modelica_real tmp3423;
  modelica_real tmp3424;
  modelica_real tmp3425;
  modelica_real tmp3426;
  modelica_real tmp3427;
  modelica_real tmp3428;
  modelica_real tmp3429;
  tmp3420 = (data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */);
  tmp3421 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3422 = (tmp3420 * tmp3420) + (tmp3421 * tmp3421);
  tmp3423 = 1.5;
  if(tmp3422 < 0.0 && tmp3423 != 0.0)
  {
    tmp3425 = modf(tmp3423, &tmp3426);
    
    if(tmp3425 > 0.5)
    {
      tmp3425 -= 1.0;
      tmp3426 += 1.0;
    }
    else if(tmp3425 < -0.5)
    {
      tmp3425 += 1.0;
      tmp3426 -= 1.0;
    }
    
    if(fabs(tmp3425) < 1e-10)
      tmp3424 = pow(tmp3422, tmp3426);
    else
    {
      tmp3428 = modf(1.0/tmp3423, &tmp3427);
      if(tmp3428 > 0.5)
      {
        tmp3428 -= 1.0;
        tmp3427 += 1.0;
      }
      else if(tmp3428 < -0.5)
      {
        tmp3428 += 1.0;
        tmp3427 -= 1.0;
      }
      if(fabs(tmp3428) < 1e-10 && ((unsigned long)tmp3427 & 1))
      {
        tmp3424 = -pow(-tmp3422, tmp3425)*pow(tmp3422, tmp3426);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3422, tmp3423);
      }
    }
  }
  else
  {
    tmp3424 = pow(tmp3422, tmp3423);
  }
  if(isnan(tmp3424) || isinf(tmp3424))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3422, tmp3423);
  }tmp3429 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3424,"(r_init[158] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3429 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[158] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3429);
    }
  }
  (data->simulationInfo->realParameter[662] /* omega_c[158] PARAM */) = sqrt(tmp3429);
  TRACE_POP
}

/*
equation index: 13687
type: SIMPLE_ASSIGN
r_init[157] = r_min + 157.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13687};
  (data->simulationInfo->realParameter[1162] /* r_init[157] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (157.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13688
type: SIMPLE_ASSIGN
omega_c[157] = sqrt(G * Md / (r_init[157] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13688(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13688};
  modelica_real tmp3430;
  modelica_real tmp3431;
  modelica_real tmp3432;
  modelica_real tmp3433;
  modelica_real tmp3434;
  modelica_real tmp3435;
  modelica_real tmp3436;
  modelica_real tmp3437;
  modelica_real tmp3438;
  modelica_real tmp3439;
  tmp3430 = (data->simulationInfo->realParameter[1162] /* r_init[157] PARAM */);
  tmp3431 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3432 = (tmp3430 * tmp3430) + (tmp3431 * tmp3431);
  tmp3433 = 1.5;
  if(tmp3432 < 0.0 && tmp3433 != 0.0)
  {
    tmp3435 = modf(tmp3433, &tmp3436);
    
    if(tmp3435 > 0.5)
    {
      tmp3435 -= 1.0;
      tmp3436 += 1.0;
    }
    else if(tmp3435 < -0.5)
    {
      tmp3435 += 1.0;
      tmp3436 -= 1.0;
    }
    
    if(fabs(tmp3435) < 1e-10)
      tmp3434 = pow(tmp3432, tmp3436);
    else
    {
      tmp3438 = modf(1.0/tmp3433, &tmp3437);
      if(tmp3438 > 0.5)
      {
        tmp3438 -= 1.0;
        tmp3437 += 1.0;
      }
      else if(tmp3438 < -0.5)
      {
        tmp3438 += 1.0;
        tmp3437 -= 1.0;
      }
      if(fabs(tmp3438) < 1e-10 && ((unsigned long)tmp3437 & 1))
      {
        tmp3434 = -pow(-tmp3432, tmp3435)*pow(tmp3432, tmp3436);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3432, tmp3433);
      }
    }
  }
  else
  {
    tmp3434 = pow(tmp3432, tmp3433);
  }
  if(isnan(tmp3434) || isinf(tmp3434))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3432, tmp3433);
  }tmp3439 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3434,"(r_init[157] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3439 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[157] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3439);
    }
  }
  (data->simulationInfo->realParameter[661] /* omega_c[157] PARAM */) = sqrt(tmp3439);
  TRACE_POP
}

/*
equation index: 13689
type: SIMPLE_ASSIGN
r_init[156] = r_min + 156.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13689};
  (data->simulationInfo->realParameter[1161] /* r_init[156] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (156.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13690
type: SIMPLE_ASSIGN
omega_c[156] = sqrt(G * Md / (r_init[156] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13690(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13690};
  modelica_real tmp3440;
  modelica_real tmp3441;
  modelica_real tmp3442;
  modelica_real tmp3443;
  modelica_real tmp3444;
  modelica_real tmp3445;
  modelica_real tmp3446;
  modelica_real tmp3447;
  modelica_real tmp3448;
  modelica_real tmp3449;
  tmp3440 = (data->simulationInfo->realParameter[1161] /* r_init[156] PARAM */);
  tmp3441 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3442 = (tmp3440 * tmp3440) + (tmp3441 * tmp3441);
  tmp3443 = 1.5;
  if(tmp3442 < 0.0 && tmp3443 != 0.0)
  {
    tmp3445 = modf(tmp3443, &tmp3446);
    
    if(tmp3445 > 0.5)
    {
      tmp3445 -= 1.0;
      tmp3446 += 1.0;
    }
    else if(tmp3445 < -0.5)
    {
      tmp3445 += 1.0;
      tmp3446 -= 1.0;
    }
    
    if(fabs(tmp3445) < 1e-10)
      tmp3444 = pow(tmp3442, tmp3446);
    else
    {
      tmp3448 = modf(1.0/tmp3443, &tmp3447);
      if(tmp3448 > 0.5)
      {
        tmp3448 -= 1.0;
        tmp3447 += 1.0;
      }
      else if(tmp3448 < -0.5)
      {
        tmp3448 += 1.0;
        tmp3447 -= 1.0;
      }
      if(fabs(tmp3448) < 1e-10 && ((unsigned long)tmp3447 & 1))
      {
        tmp3444 = -pow(-tmp3442, tmp3445)*pow(tmp3442, tmp3446);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3442, tmp3443);
      }
    }
  }
  else
  {
    tmp3444 = pow(tmp3442, tmp3443);
  }
  if(isnan(tmp3444) || isinf(tmp3444))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3442, tmp3443);
  }tmp3449 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3444,"(r_init[156] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3449 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[156] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3449);
    }
  }
  (data->simulationInfo->realParameter[660] /* omega_c[156] PARAM */) = sqrt(tmp3449);
  TRACE_POP
}

/*
equation index: 13691
type: SIMPLE_ASSIGN
r_init[155] = r_min + 155.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13691};
  (data->simulationInfo->realParameter[1160] /* r_init[155] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (155.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13692
type: SIMPLE_ASSIGN
omega_c[155] = sqrt(G * Md / (r_init[155] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13692};
  modelica_real tmp3450;
  modelica_real tmp3451;
  modelica_real tmp3452;
  modelica_real tmp3453;
  modelica_real tmp3454;
  modelica_real tmp3455;
  modelica_real tmp3456;
  modelica_real tmp3457;
  modelica_real tmp3458;
  modelica_real tmp3459;
  tmp3450 = (data->simulationInfo->realParameter[1160] /* r_init[155] PARAM */);
  tmp3451 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3452 = (tmp3450 * tmp3450) + (tmp3451 * tmp3451);
  tmp3453 = 1.5;
  if(tmp3452 < 0.0 && tmp3453 != 0.0)
  {
    tmp3455 = modf(tmp3453, &tmp3456);
    
    if(tmp3455 > 0.5)
    {
      tmp3455 -= 1.0;
      tmp3456 += 1.0;
    }
    else if(tmp3455 < -0.5)
    {
      tmp3455 += 1.0;
      tmp3456 -= 1.0;
    }
    
    if(fabs(tmp3455) < 1e-10)
      tmp3454 = pow(tmp3452, tmp3456);
    else
    {
      tmp3458 = modf(1.0/tmp3453, &tmp3457);
      if(tmp3458 > 0.5)
      {
        tmp3458 -= 1.0;
        tmp3457 += 1.0;
      }
      else if(tmp3458 < -0.5)
      {
        tmp3458 += 1.0;
        tmp3457 -= 1.0;
      }
      if(fabs(tmp3458) < 1e-10 && ((unsigned long)tmp3457 & 1))
      {
        tmp3454 = -pow(-tmp3452, tmp3455)*pow(tmp3452, tmp3456);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3452, tmp3453);
      }
    }
  }
  else
  {
    tmp3454 = pow(tmp3452, tmp3453);
  }
  if(isnan(tmp3454) || isinf(tmp3454))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3452, tmp3453);
  }tmp3459 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3454,"(r_init[155] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3459 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[155] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3459);
    }
  }
  (data->simulationInfo->realParameter[659] /* omega_c[155] PARAM */) = sqrt(tmp3459);
  TRACE_POP
}

/*
equation index: 13693
type: SIMPLE_ASSIGN
r_init[154] = r_min + 154.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13693};
  (data->simulationInfo->realParameter[1159] /* r_init[154] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (154.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13694
type: SIMPLE_ASSIGN
omega_c[154] = sqrt(G * Md / (r_init[154] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13694(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13694};
  modelica_real tmp3460;
  modelica_real tmp3461;
  modelica_real tmp3462;
  modelica_real tmp3463;
  modelica_real tmp3464;
  modelica_real tmp3465;
  modelica_real tmp3466;
  modelica_real tmp3467;
  modelica_real tmp3468;
  modelica_real tmp3469;
  tmp3460 = (data->simulationInfo->realParameter[1159] /* r_init[154] PARAM */);
  tmp3461 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3462 = (tmp3460 * tmp3460) + (tmp3461 * tmp3461);
  tmp3463 = 1.5;
  if(tmp3462 < 0.0 && tmp3463 != 0.0)
  {
    tmp3465 = modf(tmp3463, &tmp3466);
    
    if(tmp3465 > 0.5)
    {
      tmp3465 -= 1.0;
      tmp3466 += 1.0;
    }
    else if(tmp3465 < -0.5)
    {
      tmp3465 += 1.0;
      tmp3466 -= 1.0;
    }
    
    if(fabs(tmp3465) < 1e-10)
      tmp3464 = pow(tmp3462, tmp3466);
    else
    {
      tmp3468 = modf(1.0/tmp3463, &tmp3467);
      if(tmp3468 > 0.5)
      {
        tmp3468 -= 1.0;
        tmp3467 += 1.0;
      }
      else if(tmp3468 < -0.5)
      {
        tmp3468 += 1.0;
        tmp3467 -= 1.0;
      }
      if(fabs(tmp3468) < 1e-10 && ((unsigned long)tmp3467 & 1))
      {
        tmp3464 = -pow(-tmp3462, tmp3465)*pow(tmp3462, tmp3466);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3462, tmp3463);
      }
    }
  }
  else
  {
    tmp3464 = pow(tmp3462, tmp3463);
  }
  if(isnan(tmp3464) || isinf(tmp3464))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3462, tmp3463);
  }tmp3469 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3464,"(r_init[154] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3469 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[154] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3469);
    }
  }
  (data->simulationInfo->realParameter[658] /* omega_c[154] PARAM */) = sqrt(tmp3469);
  TRACE_POP
}

/*
equation index: 13695
type: SIMPLE_ASSIGN
r_init[153] = r_min + 153.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13695};
  (data->simulationInfo->realParameter[1158] /* r_init[153] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (153.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13696
type: SIMPLE_ASSIGN
omega_c[153] = sqrt(G * Md / (r_init[153] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13696(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13696};
  modelica_real tmp3470;
  modelica_real tmp3471;
  modelica_real tmp3472;
  modelica_real tmp3473;
  modelica_real tmp3474;
  modelica_real tmp3475;
  modelica_real tmp3476;
  modelica_real tmp3477;
  modelica_real tmp3478;
  modelica_real tmp3479;
  tmp3470 = (data->simulationInfo->realParameter[1158] /* r_init[153] PARAM */);
  tmp3471 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3472 = (tmp3470 * tmp3470) + (tmp3471 * tmp3471);
  tmp3473 = 1.5;
  if(tmp3472 < 0.0 && tmp3473 != 0.0)
  {
    tmp3475 = modf(tmp3473, &tmp3476);
    
    if(tmp3475 > 0.5)
    {
      tmp3475 -= 1.0;
      tmp3476 += 1.0;
    }
    else if(tmp3475 < -0.5)
    {
      tmp3475 += 1.0;
      tmp3476 -= 1.0;
    }
    
    if(fabs(tmp3475) < 1e-10)
      tmp3474 = pow(tmp3472, tmp3476);
    else
    {
      tmp3478 = modf(1.0/tmp3473, &tmp3477);
      if(tmp3478 > 0.5)
      {
        tmp3478 -= 1.0;
        tmp3477 += 1.0;
      }
      else if(tmp3478 < -0.5)
      {
        tmp3478 += 1.0;
        tmp3477 -= 1.0;
      }
      if(fabs(tmp3478) < 1e-10 && ((unsigned long)tmp3477 & 1))
      {
        tmp3474 = -pow(-tmp3472, tmp3475)*pow(tmp3472, tmp3476);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3472, tmp3473);
      }
    }
  }
  else
  {
    tmp3474 = pow(tmp3472, tmp3473);
  }
  if(isnan(tmp3474) || isinf(tmp3474))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3472, tmp3473);
  }tmp3479 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3474,"(r_init[153] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3479 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[153] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3479);
    }
  }
  (data->simulationInfo->realParameter[657] /* omega_c[153] PARAM */) = sqrt(tmp3479);
  TRACE_POP
}

/*
equation index: 13697
type: SIMPLE_ASSIGN
r_init[152] = r_min + 152.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13697};
  (data->simulationInfo->realParameter[1157] /* r_init[152] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (152.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13698
type: SIMPLE_ASSIGN
omega_c[152] = sqrt(G * Md / (r_init[152] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13698(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13698};
  modelica_real tmp3480;
  modelica_real tmp3481;
  modelica_real tmp3482;
  modelica_real tmp3483;
  modelica_real tmp3484;
  modelica_real tmp3485;
  modelica_real tmp3486;
  modelica_real tmp3487;
  modelica_real tmp3488;
  modelica_real tmp3489;
  tmp3480 = (data->simulationInfo->realParameter[1157] /* r_init[152] PARAM */);
  tmp3481 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3482 = (tmp3480 * tmp3480) + (tmp3481 * tmp3481);
  tmp3483 = 1.5;
  if(tmp3482 < 0.0 && tmp3483 != 0.0)
  {
    tmp3485 = modf(tmp3483, &tmp3486);
    
    if(tmp3485 > 0.5)
    {
      tmp3485 -= 1.0;
      tmp3486 += 1.0;
    }
    else if(tmp3485 < -0.5)
    {
      tmp3485 += 1.0;
      tmp3486 -= 1.0;
    }
    
    if(fabs(tmp3485) < 1e-10)
      tmp3484 = pow(tmp3482, tmp3486);
    else
    {
      tmp3488 = modf(1.0/tmp3483, &tmp3487);
      if(tmp3488 > 0.5)
      {
        tmp3488 -= 1.0;
        tmp3487 += 1.0;
      }
      else if(tmp3488 < -0.5)
      {
        tmp3488 += 1.0;
        tmp3487 -= 1.0;
      }
      if(fabs(tmp3488) < 1e-10 && ((unsigned long)tmp3487 & 1))
      {
        tmp3484 = -pow(-tmp3482, tmp3485)*pow(tmp3482, tmp3486);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3482, tmp3483);
      }
    }
  }
  else
  {
    tmp3484 = pow(tmp3482, tmp3483);
  }
  if(isnan(tmp3484) || isinf(tmp3484))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3482, tmp3483);
  }tmp3489 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3484,"(r_init[152] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3489 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[152] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3489);
    }
  }
  (data->simulationInfo->realParameter[656] /* omega_c[152] PARAM */) = sqrt(tmp3489);
  TRACE_POP
}

/*
equation index: 13699
type: SIMPLE_ASSIGN
r_init[151] = r_min + 151.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13699};
  (data->simulationInfo->realParameter[1156] /* r_init[151] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (151.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13700
type: SIMPLE_ASSIGN
omega_c[151] = sqrt(G * Md / (r_init[151] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13700};
  modelica_real tmp3490;
  modelica_real tmp3491;
  modelica_real tmp3492;
  modelica_real tmp3493;
  modelica_real tmp3494;
  modelica_real tmp3495;
  modelica_real tmp3496;
  modelica_real tmp3497;
  modelica_real tmp3498;
  modelica_real tmp3499;
  tmp3490 = (data->simulationInfo->realParameter[1156] /* r_init[151] PARAM */);
  tmp3491 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3492 = (tmp3490 * tmp3490) + (tmp3491 * tmp3491);
  tmp3493 = 1.5;
  if(tmp3492 < 0.0 && tmp3493 != 0.0)
  {
    tmp3495 = modf(tmp3493, &tmp3496);
    
    if(tmp3495 > 0.5)
    {
      tmp3495 -= 1.0;
      tmp3496 += 1.0;
    }
    else if(tmp3495 < -0.5)
    {
      tmp3495 += 1.0;
      tmp3496 -= 1.0;
    }
    
    if(fabs(tmp3495) < 1e-10)
      tmp3494 = pow(tmp3492, tmp3496);
    else
    {
      tmp3498 = modf(1.0/tmp3493, &tmp3497);
      if(tmp3498 > 0.5)
      {
        tmp3498 -= 1.0;
        tmp3497 += 1.0;
      }
      else if(tmp3498 < -0.5)
      {
        tmp3498 += 1.0;
        tmp3497 -= 1.0;
      }
      if(fabs(tmp3498) < 1e-10 && ((unsigned long)tmp3497 & 1))
      {
        tmp3494 = -pow(-tmp3492, tmp3495)*pow(tmp3492, tmp3496);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3492, tmp3493);
      }
    }
  }
  else
  {
    tmp3494 = pow(tmp3492, tmp3493);
  }
  if(isnan(tmp3494) || isinf(tmp3494))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3492, tmp3493);
  }tmp3499 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3494,"(r_init[151] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3499 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[151] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3499);
    }
  }
  (data->simulationInfo->realParameter[655] /* omega_c[151] PARAM */) = sqrt(tmp3499);
  TRACE_POP
}

/*
equation index: 13701
type: SIMPLE_ASSIGN
r_init[150] = r_min + 150.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13701};
  (data->simulationInfo->realParameter[1155] /* r_init[150] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (150.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13702
type: SIMPLE_ASSIGN
omega_c[150] = sqrt(G * Md / (r_init[150] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13702(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13702};
  modelica_real tmp3500;
  modelica_real tmp3501;
  modelica_real tmp3502;
  modelica_real tmp3503;
  modelica_real tmp3504;
  modelica_real tmp3505;
  modelica_real tmp3506;
  modelica_real tmp3507;
  modelica_real tmp3508;
  modelica_real tmp3509;
  tmp3500 = (data->simulationInfo->realParameter[1155] /* r_init[150] PARAM */);
  tmp3501 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3502 = (tmp3500 * tmp3500) + (tmp3501 * tmp3501);
  tmp3503 = 1.5;
  if(tmp3502 < 0.0 && tmp3503 != 0.0)
  {
    tmp3505 = modf(tmp3503, &tmp3506);
    
    if(tmp3505 > 0.5)
    {
      tmp3505 -= 1.0;
      tmp3506 += 1.0;
    }
    else if(tmp3505 < -0.5)
    {
      tmp3505 += 1.0;
      tmp3506 -= 1.0;
    }
    
    if(fabs(tmp3505) < 1e-10)
      tmp3504 = pow(tmp3502, tmp3506);
    else
    {
      tmp3508 = modf(1.0/tmp3503, &tmp3507);
      if(tmp3508 > 0.5)
      {
        tmp3508 -= 1.0;
        tmp3507 += 1.0;
      }
      else if(tmp3508 < -0.5)
      {
        tmp3508 += 1.0;
        tmp3507 -= 1.0;
      }
      if(fabs(tmp3508) < 1e-10 && ((unsigned long)tmp3507 & 1))
      {
        tmp3504 = -pow(-tmp3502, tmp3505)*pow(tmp3502, tmp3506);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3502, tmp3503);
      }
    }
  }
  else
  {
    tmp3504 = pow(tmp3502, tmp3503);
  }
  if(isnan(tmp3504) || isinf(tmp3504))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3502, tmp3503);
  }tmp3509 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3504,"(r_init[150] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3509 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[150] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3509);
    }
  }
  (data->simulationInfo->realParameter[654] /* omega_c[150] PARAM */) = sqrt(tmp3509);
  TRACE_POP
}

/*
equation index: 13703
type: SIMPLE_ASSIGN
r_init[149] = r_min + 149.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13703};
  (data->simulationInfo->realParameter[1154] /* r_init[149] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (149.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13704
type: SIMPLE_ASSIGN
omega_c[149] = sqrt(G * Md / (r_init[149] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13704(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13704};
  modelica_real tmp3510;
  modelica_real tmp3511;
  modelica_real tmp3512;
  modelica_real tmp3513;
  modelica_real tmp3514;
  modelica_real tmp3515;
  modelica_real tmp3516;
  modelica_real tmp3517;
  modelica_real tmp3518;
  modelica_real tmp3519;
  tmp3510 = (data->simulationInfo->realParameter[1154] /* r_init[149] PARAM */);
  tmp3511 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3512 = (tmp3510 * tmp3510) + (tmp3511 * tmp3511);
  tmp3513 = 1.5;
  if(tmp3512 < 0.0 && tmp3513 != 0.0)
  {
    tmp3515 = modf(tmp3513, &tmp3516);
    
    if(tmp3515 > 0.5)
    {
      tmp3515 -= 1.0;
      tmp3516 += 1.0;
    }
    else if(tmp3515 < -0.5)
    {
      tmp3515 += 1.0;
      tmp3516 -= 1.0;
    }
    
    if(fabs(tmp3515) < 1e-10)
      tmp3514 = pow(tmp3512, tmp3516);
    else
    {
      tmp3518 = modf(1.0/tmp3513, &tmp3517);
      if(tmp3518 > 0.5)
      {
        tmp3518 -= 1.0;
        tmp3517 += 1.0;
      }
      else if(tmp3518 < -0.5)
      {
        tmp3518 += 1.0;
        tmp3517 -= 1.0;
      }
      if(fabs(tmp3518) < 1e-10 && ((unsigned long)tmp3517 & 1))
      {
        tmp3514 = -pow(-tmp3512, tmp3515)*pow(tmp3512, tmp3516);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3512, tmp3513);
      }
    }
  }
  else
  {
    tmp3514 = pow(tmp3512, tmp3513);
  }
  if(isnan(tmp3514) || isinf(tmp3514))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3512, tmp3513);
  }tmp3519 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3514,"(r_init[149] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3519 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[149] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3519);
    }
  }
  (data->simulationInfo->realParameter[653] /* omega_c[149] PARAM */) = sqrt(tmp3519);
  TRACE_POP
}

/*
equation index: 13705
type: SIMPLE_ASSIGN
r_init[148] = r_min + 148.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13705};
  (data->simulationInfo->realParameter[1153] /* r_init[148] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (148.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13706
type: SIMPLE_ASSIGN
omega_c[148] = sqrt(G * Md / (r_init[148] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13706(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13706};
  modelica_real tmp3520;
  modelica_real tmp3521;
  modelica_real tmp3522;
  modelica_real tmp3523;
  modelica_real tmp3524;
  modelica_real tmp3525;
  modelica_real tmp3526;
  modelica_real tmp3527;
  modelica_real tmp3528;
  modelica_real tmp3529;
  tmp3520 = (data->simulationInfo->realParameter[1153] /* r_init[148] PARAM */);
  tmp3521 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3522 = (tmp3520 * tmp3520) + (tmp3521 * tmp3521);
  tmp3523 = 1.5;
  if(tmp3522 < 0.0 && tmp3523 != 0.0)
  {
    tmp3525 = modf(tmp3523, &tmp3526);
    
    if(tmp3525 > 0.5)
    {
      tmp3525 -= 1.0;
      tmp3526 += 1.0;
    }
    else if(tmp3525 < -0.5)
    {
      tmp3525 += 1.0;
      tmp3526 -= 1.0;
    }
    
    if(fabs(tmp3525) < 1e-10)
      tmp3524 = pow(tmp3522, tmp3526);
    else
    {
      tmp3528 = modf(1.0/tmp3523, &tmp3527);
      if(tmp3528 > 0.5)
      {
        tmp3528 -= 1.0;
        tmp3527 += 1.0;
      }
      else if(tmp3528 < -0.5)
      {
        tmp3528 += 1.0;
        tmp3527 -= 1.0;
      }
      if(fabs(tmp3528) < 1e-10 && ((unsigned long)tmp3527 & 1))
      {
        tmp3524 = -pow(-tmp3522, tmp3525)*pow(tmp3522, tmp3526);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3522, tmp3523);
      }
    }
  }
  else
  {
    tmp3524 = pow(tmp3522, tmp3523);
  }
  if(isnan(tmp3524) || isinf(tmp3524))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3522, tmp3523);
  }tmp3529 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3524,"(r_init[148] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3529 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[148] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3529);
    }
  }
  (data->simulationInfo->realParameter[652] /* omega_c[148] PARAM */) = sqrt(tmp3529);
  TRACE_POP
}

/*
equation index: 13707
type: SIMPLE_ASSIGN
r_init[147] = r_min + 147.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13707};
  (data->simulationInfo->realParameter[1152] /* r_init[147] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (147.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13708
type: SIMPLE_ASSIGN
omega_c[147] = sqrt(G * Md / (r_init[147] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13708};
  modelica_real tmp3530;
  modelica_real tmp3531;
  modelica_real tmp3532;
  modelica_real tmp3533;
  modelica_real tmp3534;
  modelica_real tmp3535;
  modelica_real tmp3536;
  modelica_real tmp3537;
  modelica_real tmp3538;
  modelica_real tmp3539;
  tmp3530 = (data->simulationInfo->realParameter[1152] /* r_init[147] PARAM */);
  tmp3531 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3532 = (tmp3530 * tmp3530) + (tmp3531 * tmp3531);
  tmp3533 = 1.5;
  if(tmp3532 < 0.0 && tmp3533 != 0.0)
  {
    tmp3535 = modf(tmp3533, &tmp3536);
    
    if(tmp3535 > 0.5)
    {
      tmp3535 -= 1.0;
      tmp3536 += 1.0;
    }
    else if(tmp3535 < -0.5)
    {
      tmp3535 += 1.0;
      tmp3536 -= 1.0;
    }
    
    if(fabs(tmp3535) < 1e-10)
      tmp3534 = pow(tmp3532, tmp3536);
    else
    {
      tmp3538 = modf(1.0/tmp3533, &tmp3537);
      if(tmp3538 > 0.5)
      {
        tmp3538 -= 1.0;
        tmp3537 += 1.0;
      }
      else if(tmp3538 < -0.5)
      {
        tmp3538 += 1.0;
        tmp3537 -= 1.0;
      }
      if(fabs(tmp3538) < 1e-10 && ((unsigned long)tmp3537 & 1))
      {
        tmp3534 = -pow(-tmp3532, tmp3535)*pow(tmp3532, tmp3536);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3532, tmp3533);
      }
    }
  }
  else
  {
    tmp3534 = pow(tmp3532, tmp3533);
  }
  if(isnan(tmp3534) || isinf(tmp3534))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3532, tmp3533);
  }tmp3539 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3534,"(r_init[147] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3539 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[147] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3539);
    }
  }
  (data->simulationInfo->realParameter[651] /* omega_c[147] PARAM */) = sqrt(tmp3539);
  TRACE_POP
}

/*
equation index: 13709
type: SIMPLE_ASSIGN
r_init[146] = r_min + 146.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13709};
  (data->simulationInfo->realParameter[1151] /* r_init[146] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (146.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13710
type: SIMPLE_ASSIGN
omega_c[146] = sqrt(G * Md / (r_init[146] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13710(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13710};
  modelica_real tmp3540;
  modelica_real tmp3541;
  modelica_real tmp3542;
  modelica_real tmp3543;
  modelica_real tmp3544;
  modelica_real tmp3545;
  modelica_real tmp3546;
  modelica_real tmp3547;
  modelica_real tmp3548;
  modelica_real tmp3549;
  tmp3540 = (data->simulationInfo->realParameter[1151] /* r_init[146] PARAM */);
  tmp3541 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3542 = (tmp3540 * tmp3540) + (tmp3541 * tmp3541);
  tmp3543 = 1.5;
  if(tmp3542 < 0.0 && tmp3543 != 0.0)
  {
    tmp3545 = modf(tmp3543, &tmp3546);
    
    if(tmp3545 > 0.5)
    {
      tmp3545 -= 1.0;
      tmp3546 += 1.0;
    }
    else if(tmp3545 < -0.5)
    {
      tmp3545 += 1.0;
      tmp3546 -= 1.0;
    }
    
    if(fabs(tmp3545) < 1e-10)
      tmp3544 = pow(tmp3542, tmp3546);
    else
    {
      tmp3548 = modf(1.0/tmp3543, &tmp3547);
      if(tmp3548 > 0.5)
      {
        tmp3548 -= 1.0;
        tmp3547 += 1.0;
      }
      else if(tmp3548 < -0.5)
      {
        tmp3548 += 1.0;
        tmp3547 -= 1.0;
      }
      if(fabs(tmp3548) < 1e-10 && ((unsigned long)tmp3547 & 1))
      {
        tmp3544 = -pow(-tmp3542, tmp3545)*pow(tmp3542, tmp3546);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3542, tmp3543);
      }
    }
  }
  else
  {
    tmp3544 = pow(tmp3542, tmp3543);
  }
  if(isnan(tmp3544) || isinf(tmp3544))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3542, tmp3543);
  }tmp3549 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3544,"(r_init[146] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3549 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[146] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3549);
    }
  }
  (data->simulationInfo->realParameter[650] /* omega_c[146] PARAM */) = sqrt(tmp3549);
  TRACE_POP
}

/*
equation index: 13711
type: SIMPLE_ASSIGN
r_init[145] = r_min + 145.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13711};
  (data->simulationInfo->realParameter[1150] /* r_init[145] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (145.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13712
type: SIMPLE_ASSIGN
omega_c[145] = sqrt(G * Md / (r_init[145] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13712(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13712};
  modelica_real tmp3550;
  modelica_real tmp3551;
  modelica_real tmp3552;
  modelica_real tmp3553;
  modelica_real tmp3554;
  modelica_real tmp3555;
  modelica_real tmp3556;
  modelica_real tmp3557;
  modelica_real tmp3558;
  modelica_real tmp3559;
  tmp3550 = (data->simulationInfo->realParameter[1150] /* r_init[145] PARAM */);
  tmp3551 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3552 = (tmp3550 * tmp3550) + (tmp3551 * tmp3551);
  tmp3553 = 1.5;
  if(tmp3552 < 0.0 && tmp3553 != 0.0)
  {
    tmp3555 = modf(tmp3553, &tmp3556);
    
    if(tmp3555 > 0.5)
    {
      tmp3555 -= 1.0;
      tmp3556 += 1.0;
    }
    else if(tmp3555 < -0.5)
    {
      tmp3555 += 1.0;
      tmp3556 -= 1.0;
    }
    
    if(fabs(tmp3555) < 1e-10)
      tmp3554 = pow(tmp3552, tmp3556);
    else
    {
      tmp3558 = modf(1.0/tmp3553, &tmp3557);
      if(tmp3558 > 0.5)
      {
        tmp3558 -= 1.0;
        tmp3557 += 1.0;
      }
      else if(tmp3558 < -0.5)
      {
        tmp3558 += 1.0;
        tmp3557 -= 1.0;
      }
      if(fabs(tmp3558) < 1e-10 && ((unsigned long)tmp3557 & 1))
      {
        tmp3554 = -pow(-tmp3552, tmp3555)*pow(tmp3552, tmp3556);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3552, tmp3553);
      }
    }
  }
  else
  {
    tmp3554 = pow(tmp3552, tmp3553);
  }
  if(isnan(tmp3554) || isinf(tmp3554))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3552, tmp3553);
  }tmp3559 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3554,"(r_init[145] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3559 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[145] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3559);
    }
  }
  (data->simulationInfo->realParameter[649] /* omega_c[145] PARAM */) = sqrt(tmp3559);
  TRACE_POP
}

/*
equation index: 13713
type: SIMPLE_ASSIGN
r_init[144] = r_min + 144.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13713};
  (data->simulationInfo->realParameter[1149] /* r_init[144] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (144.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13714
type: SIMPLE_ASSIGN
omega_c[144] = sqrt(G * Md / (r_init[144] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13714(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13714};
  modelica_real tmp3560;
  modelica_real tmp3561;
  modelica_real tmp3562;
  modelica_real tmp3563;
  modelica_real tmp3564;
  modelica_real tmp3565;
  modelica_real tmp3566;
  modelica_real tmp3567;
  modelica_real tmp3568;
  modelica_real tmp3569;
  tmp3560 = (data->simulationInfo->realParameter[1149] /* r_init[144] PARAM */);
  tmp3561 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3562 = (tmp3560 * tmp3560) + (tmp3561 * tmp3561);
  tmp3563 = 1.5;
  if(tmp3562 < 0.0 && tmp3563 != 0.0)
  {
    tmp3565 = modf(tmp3563, &tmp3566);
    
    if(tmp3565 > 0.5)
    {
      tmp3565 -= 1.0;
      tmp3566 += 1.0;
    }
    else if(tmp3565 < -0.5)
    {
      tmp3565 += 1.0;
      tmp3566 -= 1.0;
    }
    
    if(fabs(tmp3565) < 1e-10)
      tmp3564 = pow(tmp3562, tmp3566);
    else
    {
      tmp3568 = modf(1.0/tmp3563, &tmp3567);
      if(tmp3568 > 0.5)
      {
        tmp3568 -= 1.0;
        tmp3567 += 1.0;
      }
      else if(tmp3568 < -0.5)
      {
        tmp3568 += 1.0;
        tmp3567 -= 1.0;
      }
      if(fabs(tmp3568) < 1e-10 && ((unsigned long)tmp3567 & 1))
      {
        tmp3564 = -pow(-tmp3562, tmp3565)*pow(tmp3562, tmp3566);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3562, tmp3563);
      }
    }
  }
  else
  {
    tmp3564 = pow(tmp3562, tmp3563);
  }
  if(isnan(tmp3564) || isinf(tmp3564))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3562, tmp3563);
  }tmp3569 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3564,"(r_init[144] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3569 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[144] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3569);
    }
  }
  (data->simulationInfo->realParameter[648] /* omega_c[144] PARAM */) = sqrt(tmp3569);
  TRACE_POP
}

/*
equation index: 13715
type: SIMPLE_ASSIGN
r_init[143] = r_min + 143.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13715};
  (data->simulationInfo->realParameter[1148] /* r_init[143] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (143.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13716
type: SIMPLE_ASSIGN
omega_c[143] = sqrt(G * Md / (r_init[143] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13716};
  modelica_real tmp3570;
  modelica_real tmp3571;
  modelica_real tmp3572;
  modelica_real tmp3573;
  modelica_real tmp3574;
  modelica_real tmp3575;
  modelica_real tmp3576;
  modelica_real tmp3577;
  modelica_real tmp3578;
  modelica_real tmp3579;
  tmp3570 = (data->simulationInfo->realParameter[1148] /* r_init[143] PARAM */);
  tmp3571 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3572 = (tmp3570 * tmp3570) + (tmp3571 * tmp3571);
  tmp3573 = 1.5;
  if(tmp3572 < 0.0 && tmp3573 != 0.0)
  {
    tmp3575 = modf(tmp3573, &tmp3576);
    
    if(tmp3575 > 0.5)
    {
      tmp3575 -= 1.0;
      tmp3576 += 1.0;
    }
    else if(tmp3575 < -0.5)
    {
      tmp3575 += 1.0;
      tmp3576 -= 1.0;
    }
    
    if(fabs(tmp3575) < 1e-10)
      tmp3574 = pow(tmp3572, tmp3576);
    else
    {
      tmp3578 = modf(1.0/tmp3573, &tmp3577);
      if(tmp3578 > 0.5)
      {
        tmp3578 -= 1.0;
        tmp3577 += 1.0;
      }
      else if(tmp3578 < -0.5)
      {
        tmp3578 += 1.0;
        tmp3577 -= 1.0;
      }
      if(fabs(tmp3578) < 1e-10 && ((unsigned long)tmp3577 & 1))
      {
        tmp3574 = -pow(-tmp3572, tmp3575)*pow(tmp3572, tmp3576);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3572, tmp3573);
      }
    }
  }
  else
  {
    tmp3574 = pow(tmp3572, tmp3573);
  }
  if(isnan(tmp3574) || isinf(tmp3574))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3572, tmp3573);
  }tmp3579 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3574,"(r_init[143] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3579 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[143] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3579);
    }
  }
  (data->simulationInfo->realParameter[647] /* omega_c[143] PARAM */) = sqrt(tmp3579);
  TRACE_POP
}

/*
equation index: 13717
type: SIMPLE_ASSIGN
r_init[142] = r_min + 142.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13717};
  (data->simulationInfo->realParameter[1147] /* r_init[142] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (142.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13718
type: SIMPLE_ASSIGN
omega_c[142] = sqrt(G * Md / (r_init[142] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13718(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13718};
  modelica_real tmp3580;
  modelica_real tmp3581;
  modelica_real tmp3582;
  modelica_real tmp3583;
  modelica_real tmp3584;
  modelica_real tmp3585;
  modelica_real tmp3586;
  modelica_real tmp3587;
  modelica_real tmp3588;
  modelica_real tmp3589;
  tmp3580 = (data->simulationInfo->realParameter[1147] /* r_init[142] PARAM */);
  tmp3581 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3582 = (tmp3580 * tmp3580) + (tmp3581 * tmp3581);
  tmp3583 = 1.5;
  if(tmp3582 < 0.0 && tmp3583 != 0.0)
  {
    tmp3585 = modf(tmp3583, &tmp3586);
    
    if(tmp3585 > 0.5)
    {
      tmp3585 -= 1.0;
      tmp3586 += 1.0;
    }
    else if(tmp3585 < -0.5)
    {
      tmp3585 += 1.0;
      tmp3586 -= 1.0;
    }
    
    if(fabs(tmp3585) < 1e-10)
      tmp3584 = pow(tmp3582, tmp3586);
    else
    {
      tmp3588 = modf(1.0/tmp3583, &tmp3587);
      if(tmp3588 > 0.5)
      {
        tmp3588 -= 1.0;
        tmp3587 += 1.0;
      }
      else if(tmp3588 < -0.5)
      {
        tmp3588 += 1.0;
        tmp3587 -= 1.0;
      }
      if(fabs(tmp3588) < 1e-10 && ((unsigned long)tmp3587 & 1))
      {
        tmp3584 = -pow(-tmp3582, tmp3585)*pow(tmp3582, tmp3586);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3582, tmp3583);
      }
    }
  }
  else
  {
    tmp3584 = pow(tmp3582, tmp3583);
  }
  if(isnan(tmp3584) || isinf(tmp3584))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3582, tmp3583);
  }tmp3589 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3584,"(r_init[142] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3589 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[142] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3589);
    }
  }
  (data->simulationInfo->realParameter[646] /* omega_c[142] PARAM */) = sqrt(tmp3589);
  TRACE_POP
}

/*
equation index: 13719
type: SIMPLE_ASSIGN
r_init[141] = r_min + 141.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13719};
  (data->simulationInfo->realParameter[1146] /* r_init[141] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (141.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13720
type: SIMPLE_ASSIGN
omega_c[141] = sqrt(G * Md / (r_init[141] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13720(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13720};
  modelica_real tmp3590;
  modelica_real tmp3591;
  modelica_real tmp3592;
  modelica_real tmp3593;
  modelica_real tmp3594;
  modelica_real tmp3595;
  modelica_real tmp3596;
  modelica_real tmp3597;
  modelica_real tmp3598;
  modelica_real tmp3599;
  tmp3590 = (data->simulationInfo->realParameter[1146] /* r_init[141] PARAM */);
  tmp3591 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3592 = (tmp3590 * tmp3590) + (tmp3591 * tmp3591);
  tmp3593 = 1.5;
  if(tmp3592 < 0.0 && tmp3593 != 0.0)
  {
    tmp3595 = modf(tmp3593, &tmp3596);
    
    if(tmp3595 > 0.5)
    {
      tmp3595 -= 1.0;
      tmp3596 += 1.0;
    }
    else if(tmp3595 < -0.5)
    {
      tmp3595 += 1.0;
      tmp3596 -= 1.0;
    }
    
    if(fabs(tmp3595) < 1e-10)
      tmp3594 = pow(tmp3592, tmp3596);
    else
    {
      tmp3598 = modf(1.0/tmp3593, &tmp3597);
      if(tmp3598 > 0.5)
      {
        tmp3598 -= 1.0;
        tmp3597 += 1.0;
      }
      else if(tmp3598 < -0.5)
      {
        tmp3598 += 1.0;
        tmp3597 -= 1.0;
      }
      if(fabs(tmp3598) < 1e-10 && ((unsigned long)tmp3597 & 1))
      {
        tmp3594 = -pow(-tmp3592, tmp3595)*pow(tmp3592, tmp3596);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3592, tmp3593);
      }
    }
  }
  else
  {
    tmp3594 = pow(tmp3592, tmp3593);
  }
  if(isnan(tmp3594) || isinf(tmp3594))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3592, tmp3593);
  }tmp3599 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3594,"(r_init[141] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3599 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[141] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3599);
    }
  }
  (data->simulationInfo->realParameter[645] /* omega_c[141] PARAM */) = sqrt(tmp3599);
  TRACE_POP
}

/*
equation index: 13721
type: SIMPLE_ASSIGN
r_init[140] = r_min + 140.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13721};
  (data->simulationInfo->realParameter[1145] /* r_init[140] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (140.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13722
type: SIMPLE_ASSIGN
omega_c[140] = sqrt(G * Md / (r_init[140] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13722(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13722};
  modelica_real tmp3600;
  modelica_real tmp3601;
  modelica_real tmp3602;
  modelica_real tmp3603;
  modelica_real tmp3604;
  modelica_real tmp3605;
  modelica_real tmp3606;
  modelica_real tmp3607;
  modelica_real tmp3608;
  modelica_real tmp3609;
  tmp3600 = (data->simulationInfo->realParameter[1145] /* r_init[140] PARAM */);
  tmp3601 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3602 = (tmp3600 * tmp3600) + (tmp3601 * tmp3601);
  tmp3603 = 1.5;
  if(tmp3602 < 0.0 && tmp3603 != 0.0)
  {
    tmp3605 = modf(tmp3603, &tmp3606);
    
    if(tmp3605 > 0.5)
    {
      tmp3605 -= 1.0;
      tmp3606 += 1.0;
    }
    else if(tmp3605 < -0.5)
    {
      tmp3605 += 1.0;
      tmp3606 -= 1.0;
    }
    
    if(fabs(tmp3605) < 1e-10)
      tmp3604 = pow(tmp3602, tmp3606);
    else
    {
      tmp3608 = modf(1.0/tmp3603, &tmp3607);
      if(tmp3608 > 0.5)
      {
        tmp3608 -= 1.0;
        tmp3607 += 1.0;
      }
      else if(tmp3608 < -0.5)
      {
        tmp3608 += 1.0;
        tmp3607 -= 1.0;
      }
      if(fabs(tmp3608) < 1e-10 && ((unsigned long)tmp3607 & 1))
      {
        tmp3604 = -pow(-tmp3602, tmp3605)*pow(tmp3602, tmp3606);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3602, tmp3603);
      }
    }
  }
  else
  {
    tmp3604 = pow(tmp3602, tmp3603);
  }
  if(isnan(tmp3604) || isinf(tmp3604))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3602, tmp3603);
  }tmp3609 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3604,"(r_init[140] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3609 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[140] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3609);
    }
  }
  (data->simulationInfo->realParameter[644] /* omega_c[140] PARAM */) = sqrt(tmp3609);
  TRACE_POP
}

/*
equation index: 13723
type: SIMPLE_ASSIGN
r_init[139] = r_min + 139.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13723};
  (data->simulationInfo->realParameter[1144] /* r_init[139] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (139.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13724
type: SIMPLE_ASSIGN
omega_c[139] = sqrt(G * Md / (r_init[139] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13724};
  modelica_real tmp3610;
  modelica_real tmp3611;
  modelica_real tmp3612;
  modelica_real tmp3613;
  modelica_real tmp3614;
  modelica_real tmp3615;
  modelica_real tmp3616;
  modelica_real tmp3617;
  modelica_real tmp3618;
  modelica_real tmp3619;
  tmp3610 = (data->simulationInfo->realParameter[1144] /* r_init[139] PARAM */);
  tmp3611 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3612 = (tmp3610 * tmp3610) + (tmp3611 * tmp3611);
  tmp3613 = 1.5;
  if(tmp3612 < 0.0 && tmp3613 != 0.0)
  {
    tmp3615 = modf(tmp3613, &tmp3616);
    
    if(tmp3615 > 0.5)
    {
      tmp3615 -= 1.0;
      tmp3616 += 1.0;
    }
    else if(tmp3615 < -0.5)
    {
      tmp3615 += 1.0;
      tmp3616 -= 1.0;
    }
    
    if(fabs(tmp3615) < 1e-10)
      tmp3614 = pow(tmp3612, tmp3616);
    else
    {
      tmp3618 = modf(1.0/tmp3613, &tmp3617);
      if(tmp3618 > 0.5)
      {
        tmp3618 -= 1.0;
        tmp3617 += 1.0;
      }
      else if(tmp3618 < -0.5)
      {
        tmp3618 += 1.0;
        tmp3617 -= 1.0;
      }
      if(fabs(tmp3618) < 1e-10 && ((unsigned long)tmp3617 & 1))
      {
        tmp3614 = -pow(-tmp3612, tmp3615)*pow(tmp3612, tmp3616);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3612, tmp3613);
      }
    }
  }
  else
  {
    tmp3614 = pow(tmp3612, tmp3613);
  }
  if(isnan(tmp3614) || isinf(tmp3614))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3612, tmp3613);
  }tmp3619 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3614,"(r_init[139] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3619 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[139] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3619);
    }
  }
  (data->simulationInfo->realParameter[643] /* omega_c[139] PARAM */) = sqrt(tmp3619);
  TRACE_POP
}

/*
equation index: 13725
type: SIMPLE_ASSIGN
r_init[138] = r_min + 138.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13725};
  (data->simulationInfo->realParameter[1143] /* r_init[138] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (138.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13726
type: SIMPLE_ASSIGN
omega_c[138] = sqrt(G * Md / (r_init[138] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13726(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13726};
  modelica_real tmp3620;
  modelica_real tmp3621;
  modelica_real tmp3622;
  modelica_real tmp3623;
  modelica_real tmp3624;
  modelica_real tmp3625;
  modelica_real tmp3626;
  modelica_real tmp3627;
  modelica_real tmp3628;
  modelica_real tmp3629;
  tmp3620 = (data->simulationInfo->realParameter[1143] /* r_init[138] PARAM */);
  tmp3621 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3622 = (tmp3620 * tmp3620) + (tmp3621 * tmp3621);
  tmp3623 = 1.5;
  if(tmp3622 < 0.0 && tmp3623 != 0.0)
  {
    tmp3625 = modf(tmp3623, &tmp3626);
    
    if(tmp3625 > 0.5)
    {
      tmp3625 -= 1.0;
      tmp3626 += 1.0;
    }
    else if(tmp3625 < -0.5)
    {
      tmp3625 += 1.0;
      tmp3626 -= 1.0;
    }
    
    if(fabs(tmp3625) < 1e-10)
      tmp3624 = pow(tmp3622, tmp3626);
    else
    {
      tmp3628 = modf(1.0/tmp3623, &tmp3627);
      if(tmp3628 > 0.5)
      {
        tmp3628 -= 1.0;
        tmp3627 += 1.0;
      }
      else if(tmp3628 < -0.5)
      {
        tmp3628 += 1.0;
        tmp3627 -= 1.0;
      }
      if(fabs(tmp3628) < 1e-10 && ((unsigned long)tmp3627 & 1))
      {
        tmp3624 = -pow(-tmp3622, tmp3625)*pow(tmp3622, tmp3626);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3622, tmp3623);
      }
    }
  }
  else
  {
    tmp3624 = pow(tmp3622, tmp3623);
  }
  if(isnan(tmp3624) || isinf(tmp3624))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3622, tmp3623);
  }tmp3629 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3624,"(r_init[138] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3629 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[138] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3629);
    }
  }
  (data->simulationInfo->realParameter[642] /* omega_c[138] PARAM */) = sqrt(tmp3629);
  TRACE_POP
}

/*
equation index: 13727
type: SIMPLE_ASSIGN
r_init[137] = r_min + 137.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13727};
  (data->simulationInfo->realParameter[1142] /* r_init[137] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (137.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13728
type: SIMPLE_ASSIGN
omega_c[137] = sqrt(G * Md / (r_init[137] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13728(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13728};
  modelica_real tmp3630;
  modelica_real tmp3631;
  modelica_real tmp3632;
  modelica_real tmp3633;
  modelica_real tmp3634;
  modelica_real tmp3635;
  modelica_real tmp3636;
  modelica_real tmp3637;
  modelica_real tmp3638;
  modelica_real tmp3639;
  tmp3630 = (data->simulationInfo->realParameter[1142] /* r_init[137] PARAM */);
  tmp3631 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3632 = (tmp3630 * tmp3630) + (tmp3631 * tmp3631);
  tmp3633 = 1.5;
  if(tmp3632 < 0.0 && tmp3633 != 0.0)
  {
    tmp3635 = modf(tmp3633, &tmp3636);
    
    if(tmp3635 > 0.5)
    {
      tmp3635 -= 1.0;
      tmp3636 += 1.0;
    }
    else if(tmp3635 < -0.5)
    {
      tmp3635 += 1.0;
      tmp3636 -= 1.0;
    }
    
    if(fabs(tmp3635) < 1e-10)
      tmp3634 = pow(tmp3632, tmp3636);
    else
    {
      tmp3638 = modf(1.0/tmp3633, &tmp3637);
      if(tmp3638 > 0.5)
      {
        tmp3638 -= 1.0;
        tmp3637 += 1.0;
      }
      else if(tmp3638 < -0.5)
      {
        tmp3638 += 1.0;
        tmp3637 -= 1.0;
      }
      if(fabs(tmp3638) < 1e-10 && ((unsigned long)tmp3637 & 1))
      {
        tmp3634 = -pow(-tmp3632, tmp3635)*pow(tmp3632, tmp3636);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3632, tmp3633);
      }
    }
  }
  else
  {
    tmp3634 = pow(tmp3632, tmp3633);
  }
  if(isnan(tmp3634) || isinf(tmp3634))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3632, tmp3633);
  }tmp3639 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3634,"(r_init[137] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3639 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[137] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3639);
    }
  }
  (data->simulationInfo->realParameter[641] /* omega_c[137] PARAM */) = sqrt(tmp3639);
  TRACE_POP
}

/*
equation index: 13729
type: SIMPLE_ASSIGN
r_init[136] = r_min + 136.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13729};
  (data->simulationInfo->realParameter[1141] /* r_init[136] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (136.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13730
type: SIMPLE_ASSIGN
omega_c[136] = sqrt(G * Md / (r_init[136] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13730(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13730};
  modelica_real tmp3640;
  modelica_real tmp3641;
  modelica_real tmp3642;
  modelica_real tmp3643;
  modelica_real tmp3644;
  modelica_real tmp3645;
  modelica_real tmp3646;
  modelica_real tmp3647;
  modelica_real tmp3648;
  modelica_real tmp3649;
  tmp3640 = (data->simulationInfo->realParameter[1141] /* r_init[136] PARAM */);
  tmp3641 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3642 = (tmp3640 * tmp3640) + (tmp3641 * tmp3641);
  tmp3643 = 1.5;
  if(tmp3642 < 0.0 && tmp3643 != 0.0)
  {
    tmp3645 = modf(tmp3643, &tmp3646);
    
    if(tmp3645 > 0.5)
    {
      tmp3645 -= 1.0;
      tmp3646 += 1.0;
    }
    else if(tmp3645 < -0.5)
    {
      tmp3645 += 1.0;
      tmp3646 -= 1.0;
    }
    
    if(fabs(tmp3645) < 1e-10)
      tmp3644 = pow(tmp3642, tmp3646);
    else
    {
      tmp3648 = modf(1.0/tmp3643, &tmp3647);
      if(tmp3648 > 0.5)
      {
        tmp3648 -= 1.0;
        tmp3647 += 1.0;
      }
      else if(tmp3648 < -0.5)
      {
        tmp3648 += 1.0;
        tmp3647 -= 1.0;
      }
      if(fabs(tmp3648) < 1e-10 && ((unsigned long)tmp3647 & 1))
      {
        tmp3644 = -pow(-tmp3642, tmp3645)*pow(tmp3642, tmp3646);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3642, tmp3643);
      }
    }
  }
  else
  {
    tmp3644 = pow(tmp3642, tmp3643);
  }
  if(isnan(tmp3644) || isinf(tmp3644))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3642, tmp3643);
  }tmp3649 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3644,"(r_init[136] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3649 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[136] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3649);
    }
  }
  (data->simulationInfo->realParameter[640] /* omega_c[136] PARAM */) = sqrt(tmp3649);
  TRACE_POP
}

/*
equation index: 13731
type: SIMPLE_ASSIGN
r_init[135] = r_min + 135.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13731};
  (data->simulationInfo->realParameter[1140] /* r_init[135] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (135.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13732
type: SIMPLE_ASSIGN
omega_c[135] = sqrt(G * Md / (r_init[135] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13732};
  modelica_real tmp3650;
  modelica_real tmp3651;
  modelica_real tmp3652;
  modelica_real tmp3653;
  modelica_real tmp3654;
  modelica_real tmp3655;
  modelica_real tmp3656;
  modelica_real tmp3657;
  modelica_real tmp3658;
  modelica_real tmp3659;
  tmp3650 = (data->simulationInfo->realParameter[1140] /* r_init[135] PARAM */);
  tmp3651 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3652 = (tmp3650 * tmp3650) + (tmp3651 * tmp3651);
  tmp3653 = 1.5;
  if(tmp3652 < 0.0 && tmp3653 != 0.0)
  {
    tmp3655 = modf(tmp3653, &tmp3656);
    
    if(tmp3655 > 0.5)
    {
      tmp3655 -= 1.0;
      tmp3656 += 1.0;
    }
    else if(tmp3655 < -0.5)
    {
      tmp3655 += 1.0;
      tmp3656 -= 1.0;
    }
    
    if(fabs(tmp3655) < 1e-10)
      tmp3654 = pow(tmp3652, tmp3656);
    else
    {
      tmp3658 = modf(1.0/tmp3653, &tmp3657);
      if(tmp3658 > 0.5)
      {
        tmp3658 -= 1.0;
        tmp3657 += 1.0;
      }
      else if(tmp3658 < -0.5)
      {
        tmp3658 += 1.0;
        tmp3657 -= 1.0;
      }
      if(fabs(tmp3658) < 1e-10 && ((unsigned long)tmp3657 & 1))
      {
        tmp3654 = -pow(-tmp3652, tmp3655)*pow(tmp3652, tmp3656);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3652, tmp3653);
      }
    }
  }
  else
  {
    tmp3654 = pow(tmp3652, tmp3653);
  }
  if(isnan(tmp3654) || isinf(tmp3654))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3652, tmp3653);
  }tmp3659 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3654,"(r_init[135] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3659 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[135] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3659);
    }
  }
  (data->simulationInfo->realParameter[639] /* omega_c[135] PARAM */) = sqrt(tmp3659);
  TRACE_POP
}

/*
equation index: 13733
type: SIMPLE_ASSIGN
r_init[134] = r_min + 134.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13733};
  (data->simulationInfo->realParameter[1139] /* r_init[134] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (134.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13734
type: SIMPLE_ASSIGN
omega_c[134] = sqrt(G * Md / (r_init[134] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13734(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13734};
  modelica_real tmp3660;
  modelica_real tmp3661;
  modelica_real tmp3662;
  modelica_real tmp3663;
  modelica_real tmp3664;
  modelica_real tmp3665;
  modelica_real tmp3666;
  modelica_real tmp3667;
  modelica_real tmp3668;
  modelica_real tmp3669;
  tmp3660 = (data->simulationInfo->realParameter[1139] /* r_init[134] PARAM */);
  tmp3661 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3662 = (tmp3660 * tmp3660) + (tmp3661 * tmp3661);
  tmp3663 = 1.5;
  if(tmp3662 < 0.0 && tmp3663 != 0.0)
  {
    tmp3665 = modf(tmp3663, &tmp3666);
    
    if(tmp3665 > 0.5)
    {
      tmp3665 -= 1.0;
      tmp3666 += 1.0;
    }
    else if(tmp3665 < -0.5)
    {
      tmp3665 += 1.0;
      tmp3666 -= 1.0;
    }
    
    if(fabs(tmp3665) < 1e-10)
      tmp3664 = pow(tmp3662, tmp3666);
    else
    {
      tmp3668 = modf(1.0/tmp3663, &tmp3667);
      if(tmp3668 > 0.5)
      {
        tmp3668 -= 1.0;
        tmp3667 += 1.0;
      }
      else if(tmp3668 < -0.5)
      {
        tmp3668 += 1.0;
        tmp3667 -= 1.0;
      }
      if(fabs(tmp3668) < 1e-10 && ((unsigned long)tmp3667 & 1))
      {
        tmp3664 = -pow(-tmp3662, tmp3665)*pow(tmp3662, tmp3666);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3662, tmp3663);
      }
    }
  }
  else
  {
    tmp3664 = pow(tmp3662, tmp3663);
  }
  if(isnan(tmp3664) || isinf(tmp3664))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3662, tmp3663);
  }tmp3669 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3664,"(r_init[134] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3669 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[134] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3669);
    }
  }
  (data->simulationInfo->realParameter[638] /* omega_c[134] PARAM */) = sqrt(tmp3669);
  TRACE_POP
}

/*
equation index: 13735
type: SIMPLE_ASSIGN
r_init[133] = r_min + 133.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13735};
  (data->simulationInfo->realParameter[1138] /* r_init[133] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (133.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13736
type: SIMPLE_ASSIGN
omega_c[133] = sqrt(G * Md / (r_init[133] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13736(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13736};
  modelica_real tmp3670;
  modelica_real tmp3671;
  modelica_real tmp3672;
  modelica_real tmp3673;
  modelica_real tmp3674;
  modelica_real tmp3675;
  modelica_real tmp3676;
  modelica_real tmp3677;
  modelica_real tmp3678;
  modelica_real tmp3679;
  tmp3670 = (data->simulationInfo->realParameter[1138] /* r_init[133] PARAM */);
  tmp3671 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3672 = (tmp3670 * tmp3670) + (tmp3671 * tmp3671);
  tmp3673 = 1.5;
  if(tmp3672 < 0.0 && tmp3673 != 0.0)
  {
    tmp3675 = modf(tmp3673, &tmp3676);
    
    if(tmp3675 > 0.5)
    {
      tmp3675 -= 1.0;
      tmp3676 += 1.0;
    }
    else if(tmp3675 < -0.5)
    {
      tmp3675 += 1.0;
      tmp3676 -= 1.0;
    }
    
    if(fabs(tmp3675) < 1e-10)
      tmp3674 = pow(tmp3672, tmp3676);
    else
    {
      tmp3678 = modf(1.0/tmp3673, &tmp3677);
      if(tmp3678 > 0.5)
      {
        tmp3678 -= 1.0;
        tmp3677 += 1.0;
      }
      else if(tmp3678 < -0.5)
      {
        tmp3678 += 1.0;
        tmp3677 -= 1.0;
      }
      if(fabs(tmp3678) < 1e-10 && ((unsigned long)tmp3677 & 1))
      {
        tmp3674 = -pow(-tmp3672, tmp3675)*pow(tmp3672, tmp3676);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3672, tmp3673);
      }
    }
  }
  else
  {
    tmp3674 = pow(tmp3672, tmp3673);
  }
  if(isnan(tmp3674) || isinf(tmp3674))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3672, tmp3673);
  }tmp3679 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3674,"(r_init[133] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3679 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[133] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3679);
    }
  }
  (data->simulationInfo->realParameter[637] /* omega_c[133] PARAM */) = sqrt(tmp3679);
  TRACE_POP
}

/*
equation index: 13737
type: SIMPLE_ASSIGN
r_init[132] = r_min + 132.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13737};
  (data->simulationInfo->realParameter[1137] /* r_init[132] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (132.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13738
type: SIMPLE_ASSIGN
omega_c[132] = sqrt(G * Md / (r_init[132] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13738(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13738};
  modelica_real tmp3680;
  modelica_real tmp3681;
  modelica_real tmp3682;
  modelica_real tmp3683;
  modelica_real tmp3684;
  modelica_real tmp3685;
  modelica_real tmp3686;
  modelica_real tmp3687;
  modelica_real tmp3688;
  modelica_real tmp3689;
  tmp3680 = (data->simulationInfo->realParameter[1137] /* r_init[132] PARAM */);
  tmp3681 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3682 = (tmp3680 * tmp3680) + (tmp3681 * tmp3681);
  tmp3683 = 1.5;
  if(tmp3682 < 0.0 && tmp3683 != 0.0)
  {
    tmp3685 = modf(tmp3683, &tmp3686);
    
    if(tmp3685 > 0.5)
    {
      tmp3685 -= 1.0;
      tmp3686 += 1.0;
    }
    else if(tmp3685 < -0.5)
    {
      tmp3685 += 1.0;
      tmp3686 -= 1.0;
    }
    
    if(fabs(tmp3685) < 1e-10)
      tmp3684 = pow(tmp3682, tmp3686);
    else
    {
      tmp3688 = modf(1.0/tmp3683, &tmp3687);
      if(tmp3688 > 0.5)
      {
        tmp3688 -= 1.0;
        tmp3687 += 1.0;
      }
      else if(tmp3688 < -0.5)
      {
        tmp3688 += 1.0;
        tmp3687 -= 1.0;
      }
      if(fabs(tmp3688) < 1e-10 && ((unsigned long)tmp3687 & 1))
      {
        tmp3684 = -pow(-tmp3682, tmp3685)*pow(tmp3682, tmp3686);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3682, tmp3683);
      }
    }
  }
  else
  {
    tmp3684 = pow(tmp3682, tmp3683);
  }
  if(isnan(tmp3684) || isinf(tmp3684))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3682, tmp3683);
  }tmp3689 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3684,"(r_init[132] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3689 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[132] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3689);
    }
  }
  (data->simulationInfo->realParameter[636] /* omega_c[132] PARAM */) = sqrt(tmp3689);
  TRACE_POP
}

/*
equation index: 13739
type: SIMPLE_ASSIGN
r_init[131] = r_min + 131.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13739};
  (data->simulationInfo->realParameter[1136] /* r_init[131] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (131.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13740
type: SIMPLE_ASSIGN
omega_c[131] = sqrt(G * Md / (r_init[131] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13740};
  modelica_real tmp3690;
  modelica_real tmp3691;
  modelica_real tmp3692;
  modelica_real tmp3693;
  modelica_real tmp3694;
  modelica_real tmp3695;
  modelica_real tmp3696;
  modelica_real tmp3697;
  modelica_real tmp3698;
  modelica_real tmp3699;
  tmp3690 = (data->simulationInfo->realParameter[1136] /* r_init[131] PARAM */);
  tmp3691 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3692 = (tmp3690 * tmp3690) + (tmp3691 * tmp3691);
  tmp3693 = 1.5;
  if(tmp3692 < 0.0 && tmp3693 != 0.0)
  {
    tmp3695 = modf(tmp3693, &tmp3696);
    
    if(tmp3695 > 0.5)
    {
      tmp3695 -= 1.0;
      tmp3696 += 1.0;
    }
    else if(tmp3695 < -0.5)
    {
      tmp3695 += 1.0;
      tmp3696 -= 1.0;
    }
    
    if(fabs(tmp3695) < 1e-10)
      tmp3694 = pow(tmp3692, tmp3696);
    else
    {
      tmp3698 = modf(1.0/tmp3693, &tmp3697);
      if(tmp3698 > 0.5)
      {
        tmp3698 -= 1.0;
        tmp3697 += 1.0;
      }
      else if(tmp3698 < -0.5)
      {
        tmp3698 += 1.0;
        tmp3697 -= 1.0;
      }
      if(fabs(tmp3698) < 1e-10 && ((unsigned long)tmp3697 & 1))
      {
        tmp3694 = -pow(-tmp3692, tmp3695)*pow(tmp3692, tmp3696);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3692, tmp3693);
      }
    }
  }
  else
  {
    tmp3694 = pow(tmp3692, tmp3693);
  }
  if(isnan(tmp3694) || isinf(tmp3694))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3692, tmp3693);
  }tmp3699 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3694,"(r_init[131] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3699 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[131] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3699);
    }
  }
  (data->simulationInfo->realParameter[635] /* omega_c[131] PARAM */) = sqrt(tmp3699);
  TRACE_POP
}

/*
equation index: 13741
type: SIMPLE_ASSIGN
r_init[130] = r_min + 130.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13741};
  (data->simulationInfo->realParameter[1135] /* r_init[130] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (130.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13742
type: SIMPLE_ASSIGN
omega_c[130] = sqrt(G * Md / (r_init[130] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13742(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13742};
  modelica_real tmp3700;
  modelica_real tmp3701;
  modelica_real tmp3702;
  modelica_real tmp3703;
  modelica_real tmp3704;
  modelica_real tmp3705;
  modelica_real tmp3706;
  modelica_real tmp3707;
  modelica_real tmp3708;
  modelica_real tmp3709;
  tmp3700 = (data->simulationInfo->realParameter[1135] /* r_init[130] PARAM */);
  tmp3701 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3702 = (tmp3700 * tmp3700) + (tmp3701 * tmp3701);
  tmp3703 = 1.5;
  if(tmp3702 < 0.0 && tmp3703 != 0.0)
  {
    tmp3705 = modf(tmp3703, &tmp3706);
    
    if(tmp3705 > 0.5)
    {
      tmp3705 -= 1.0;
      tmp3706 += 1.0;
    }
    else if(tmp3705 < -0.5)
    {
      tmp3705 += 1.0;
      tmp3706 -= 1.0;
    }
    
    if(fabs(tmp3705) < 1e-10)
      tmp3704 = pow(tmp3702, tmp3706);
    else
    {
      tmp3708 = modf(1.0/tmp3703, &tmp3707);
      if(tmp3708 > 0.5)
      {
        tmp3708 -= 1.0;
        tmp3707 += 1.0;
      }
      else if(tmp3708 < -0.5)
      {
        tmp3708 += 1.0;
        tmp3707 -= 1.0;
      }
      if(fabs(tmp3708) < 1e-10 && ((unsigned long)tmp3707 & 1))
      {
        tmp3704 = -pow(-tmp3702, tmp3705)*pow(tmp3702, tmp3706);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3702, tmp3703);
      }
    }
  }
  else
  {
    tmp3704 = pow(tmp3702, tmp3703);
  }
  if(isnan(tmp3704) || isinf(tmp3704))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3702, tmp3703);
  }tmp3709 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3704,"(r_init[130] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3709 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[130] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3709);
    }
  }
  (data->simulationInfo->realParameter[634] /* omega_c[130] PARAM */) = sqrt(tmp3709);
  TRACE_POP
}

/*
equation index: 13743
type: SIMPLE_ASSIGN
r_init[129] = r_min + 129.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13743};
  (data->simulationInfo->realParameter[1134] /* r_init[129] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (129.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13744
type: SIMPLE_ASSIGN
omega_c[129] = sqrt(G * Md / (r_init[129] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13744(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13744};
  modelica_real tmp3710;
  modelica_real tmp3711;
  modelica_real tmp3712;
  modelica_real tmp3713;
  modelica_real tmp3714;
  modelica_real tmp3715;
  modelica_real tmp3716;
  modelica_real tmp3717;
  modelica_real tmp3718;
  modelica_real tmp3719;
  tmp3710 = (data->simulationInfo->realParameter[1134] /* r_init[129] PARAM */);
  tmp3711 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3712 = (tmp3710 * tmp3710) + (tmp3711 * tmp3711);
  tmp3713 = 1.5;
  if(tmp3712 < 0.0 && tmp3713 != 0.0)
  {
    tmp3715 = modf(tmp3713, &tmp3716);
    
    if(tmp3715 > 0.5)
    {
      tmp3715 -= 1.0;
      tmp3716 += 1.0;
    }
    else if(tmp3715 < -0.5)
    {
      tmp3715 += 1.0;
      tmp3716 -= 1.0;
    }
    
    if(fabs(tmp3715) < 1e-10)
      tmp3714 = pow(tmp3712, tmp3716);
    else
    {
      tmp3718 = modf(1.0/tmp3713, &tmp3717);
      if(tmp3718 > 0.5)
      {
        tmp3718 -= 1.0;
        tmp3717 += 1.0;
      }
      else if(tmp3718 < -0.5)
      {
        tmp3718 += 1.0;
        tmp3717 -= 1.0;
      }
      if(fabs(tmp3718) < 1e-10 && ((unsigned long)tmp3717 & 1))
      {
        tmp3714 = -pow(-tmp3712, tmp3715)*pow(tmp3712, tmp3716);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3712, tmp3713);
      }
    }
  }
  else
  {
    tmp3714 = pow(tmp3712, tmp3713);
  }
  if(isnan(tmp3714) || isinf(tmp3714))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3712, tmp3713);
  }tmp3719 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3714,"(r_init[129] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3719 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[129] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3719);
    }
  }
  (data->simulationInfo->realParameter[633] /* omega_c[129] PARAM */) = sqrt(tmp3719);
  TRACE_POP
}

/*
equation index: 13745
type: SIMPLE_ASSIGN
r_init[128] = r_min + 128.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13745};
  (data->simulationInfo->realParameter[1133] /* r_init[128] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (128.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13746
type: SIMPLE_ASSIGN
omega_c[128] = sqrt(G * Md / (r_init[128] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13746(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13746};
  modelica_real tmp3720;
  modelica_real tmp3721;
  modelica_real tmp3722;
  modelica_real tmp3723;
  modelica_real tmp3724;
  modelica_real tmp3725;
  modelica_real tmp3726;
  modelica_real tmp3727;
  modelica_real tmp3728;
  modelica_real tmp3729;
  tmp3720 = (data->simulationInfo->realParameter[1133] /* r_init[128] PARAM */);
  tmp3721 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3722 = (tmp3720 * tmp3720) + (tmp3721 * tmp3721);
  tmp3723 = 1.5;
  if(tmp3722 < 0.0 && tmp3723 != 0.0)
  {
    tmp3725 = modf(tmp3723, &tmp3726);
    
    if(tmp3725 > 0.5)
    {
      tmp3725 -= 1.0;
      tmp3726 += 1.0;
    }
    else if(tmp3725 < -0.5)
    {
      tmp3725 += 1.0;
      tmp3726 -= 1.0;
    }
    
    if(fabs(tmp3725) < 1e-10)
      tmp3724 = pow(tmp3722, tmp3726);
    else
    {
      tmp3728 = modf(1.0/tmp3723, &tmp3727);
      if(tmp3728 > 0.5)
      {
        tmp3728 -= 1.0;
        tmp3727 += 1.0;
      }
      else if(tmp3728 < -0.5)
      {
        tmp3728 += 1.0;
        tmp3727 -= 1.0;
      }
      if(fabs(tmp3728) < 1e-10 && ((unsigned long)tmp3727 & 1))
      {
        tmp3724 = -pow(-tmp3722, tmp3725)*pow(tmp3722, tmp3726);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3722, tmp3723);
      }
    }
  }
  else
  {
    tmp3724 = pow(tmp3722, tmp3723);
  }
  if(isnan(tmp3724) || isinf(tmp3724))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3722, tmp3723);
  }tmp3729 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3724,"(r_init[128] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3729 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[128] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3729);
    }
  }
  (data->simulationInfo->realParameter[632] /* omega_c[128] PARAM */) = sqrt(tmp3729);
  TRACE_POP
}

/*
equation index: 13747
type: SIMPLE_ASSIGN
r_init[127] = r_min + 127.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13747};
  (data->simulationInfo->realParameter[1132] /* r_init[127] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (127.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13748
type: SIMPLE_ASSIGN
omega_c[127] = sqrt(G * Md / (r_init[127] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13748};
  modelica_real tmp3730;
  modelica_real tmp3731;
  modelica_real tmp3732;
  modelica_real tmp3733;
  modelica_real tmp3734;
  modelica_real tmp3735;
  modelica_real tmp3736;
  modelica_real tmp3737;
  modelica_real tmp3738;
  modelica_real tmp3739;
  tmp3730 = (data->simulationInfo->realParameter[1132] /* r_init[127] PARAM */);
  tmp3731 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3732 = (tmp3730 * tmp3730) + (tmp3731 * tmp3731);
  tmp3733 = 1.5;
  if(tmp3732 < 0.0 && tmp3733 != 0.0)
  {
    tmp3735 = modf(tmp3733, &tmp3736);
    
    if(tmp3735 > 0.5)
    {
      tmp3735 -= 1.0;
      tmp3736 += 1.0;
    }
    else if(tmp3735 < -0.5)
    {
      tmp3735 += 1.0;
      tmp3736 -= 1.0;
    }
    
    if(fabs(tmp3735) < 1e-10)
      tmp3734 = pow(tmp3732, tmp3736);
    else
    {
      tmp3738 = modf(1.0/tmp3733, &tmp3737);
      if(tmp3738 > 0.5)
      {
        tmp3738 -= 1.0;
        tmp3737 += 1.0;
      }
      else if(tmp3738 < -0.5)
      {
        tmp3738 += 1.0;
        tmp3737 -= 1.0;
      }
      if(fabs(tmp3738) < 1e-10 && ((unsigned long)tmp3737 & 1))
      {
        tmp3734 = -pow(-tmp3732, tmp3735)*pow(tmp3732, tmp3736);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3732, tmp3733);
      }
    }
  }
  else
  {
    tmp3734 = pow(tmp3732, tmp3733);
  }
  if(isnan(tmp3734) || isinf(tmp3734))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3732, tmp3733);
  }tmp3739 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3734,"(r_init[127] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3739 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[127] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3739);
    }
  }
  (data->simulationInfo->realParameter[631] /* omega_c[127] PARAM */) = sqrt(tmp3739);
  TRACE_POP
}

/*
equation index: 13749
type: SIMPLE_ASSIGN
r_init[126] = r_min + 126.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13749};
  (data->simulationInfo->realParameter[1131] /* r_init[126] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (126.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13750
type: SIMPLE_ASSIGN
omega_c[126] = sqrt(G * Md / (r_init[126] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13750(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13750};
  modelica_real tmp3740;
  modelica_real tmp3741;
  modelica_real tmp3742;
  modelica_real tmp3743;
  modelica_real tmp3744;
  modelica_real tmp3745;
  modelica_real tmp3746;
  modelica_real tmp3747;
  modelica_real tmp3748;
  modelica_real tmp3749;
  tmp3740 = (data->simulationInfo->realParameter[1131] /* r_init[126] PARAM */);
  tmp3741 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3742 = (tmp3740 * tmp3740) + (tmp3741 * tmp3741);
  tmp3743 = 1.5;
  if(tmp3742 < 0.0 && tmp3743 != 0.0)
  {
    tmp3745 = modf(tmp3743, &tmp3746);
    
    if(tmp3745 > 0.5)
    {
      tmp3745 -= 1.0;
      tmp3746 += 1.0;
    }
    else if(tmp3745 < -0.5)
    {
      tmp3745 += 1.0;
      tmp3746 -= 1.0;
    }
    
    if(fabs(tmp3745) < 1e-10)
      tmp3744 = pow(tmp3742, tmp3746);
    else
    {
      tmp3748 = modf(1.0/tmp3743, &tmp3747);
      if(tmp3748 > 0.5)
      {
        tmp3748 -= 1.0;
        tmp3747 += 1.0;
      }
      else if(tmp3748 < -0.5)
      {
        tmp3748 += 1.0;
        tmp3747 -= 1.0;
      }
      if(fabs(tmp3748) < 1e-10 && ((unsigned long)tmp3747 & 1))
      {
        tmp3744 = -pow(-tmp3742, tmp3745)*pow(tmp3742, tmp3746);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3742, tmp3743);
      }
    }
  }
  else
  {
    tmp3744 = pow(tmp3742, tmp3743);
  }
  if(isnan(tmp3744) || isinf(tmp3744))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3742, tmp3743);
  }tmp3749 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3744,"(r_init[126] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3749 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[126] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3749);
    }
  }
  (data->simulationInfo->realParameter[630] /* omega_c[126] PARAM */) = sqrt(tmp3749);
  TRACE_POP
}

/*
equation index: 13751
type: SIMPLE_ASSIGN
r_init[125] = r_min + 125.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13751};
  (data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (125.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13752
type: SIMPLE_ASSIGN
omega_c[125] = sqrt(G * Md / (r_init[125] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13752(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13752};
  modelica_real tmp3750;
  modelica_real tmp3751;
  modelica_real tmp3752;
  modelica_real tmp3753;
  modelica_real tmp3754;
  modelica_real tmp3755;
  modelica_real tmp3756;
  modelica_real tmp3757;
  modelica_real tmp3758;
  modelica_real tmp3759;
  tmp3750 = (data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */);
  tmp3751 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3752 = (tmp3750 * tmp3750) + (tmp3751 * tmp3751);
  tmp3753 = 1.5;
  if(tmp3752 < 0.0 && tmp3753 != 0.0)
  {
    tmp3755 = modf(tmp3753, &tmp3756);
    
    if(tmp3755 > 0.5)
    {
      tmp3755 -= 1.0;
      tmp3756 += 1.0;
    }
    else if(tmp3755 < -0.5)
    {
      tmp3755 += 1.0;
      tmp3756 -= 1.0;
    }
    
    if(fabs(tmp3755) < 1e-10)
      tmp3754 = pow(tmp3752, tmp3756);
    else
    {
      tmp3758 = modf(1.0/tmp3753, &tmp3757);
      if(tmp3758 > 0.5)
      {
        tmp3758 -= 1.0;
        tmp3757 += 1.0;
      }
      else if(tmp3758 < -0.5)
      {
        tmp3758 += 1.0;
        tmp3757 -= 1.0;
      }
      if(fabs(tmp3758) < 1e-10 && ((unsigned long)tmp3757 & 1))
      {
        tmp3754 = -pow(-tmp3752, tmp3755)*pow(tmp3752, tmp3756);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3752, tmp3753);
      }
    }
  }
  else
  {
    tmp3754 = pow(tmp3752, tmp3753);
  }
  if(isnan(tmp3754) || isinf(tmp3754))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3752, tmp3753);
  }tmp3759 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3754,"(r_init[125] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3759 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[125] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3759);
    }
  }
  (data->simulationInfo->realParameter[629] /* omega_c[125] PARAM */) = sqrt(tmp3759);
  TRACE_POP
}

/*
equation index: 13753
type: SIMPLE_ASSIGN
r_init[124] = r_min + 124.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13753};
  (data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (124.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13754
type: SIMPLE_ASSIGN
omega_c[124] = sqrt(G * Md / (r_init[124] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13754(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13754};
  modelica_real tmp3760;
  modelica_real tmp3761;
  modelica_real tmp3762;
  modelica_real tmp3763;
  modelica_real tmp3764;
  modelica_real tmp3765;
  modelica_real tmp3766;
  modelica_real tmp3767;
  modelica_real tmp3768;
  modelica_real tmp3769;
  tmp3760 = (data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */);
  tmp3761 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3762 = (tmp3760 * tmp3760) + (tmp3761 * tmp3761);
  tmp3763 = 1.5;
  if(tmp3762 < 0.0 && tmp3763 != 0.0)
  {
    tmp3765 = modf(tmp3763, &tmp3766);
    
    if(tmp3765 > 0.5)
    {
      tmp3765 -= 1.0;
      tmp3766 += 1.0;
    }
    else if(tmp3765 < -0.5)
    {
      tmp3765 += 1.0;
      tmp3766 -= 1.0;
    }
    
    if(fabs(tmp3765) < 1e-10)
      tmp3764 = pow(tmp3762, tmp3766);
    else
    {
      tmp3768 = modf(1.0/tmp3763, &tmp3767);
      if(tmp3768 > 0.5)
      {
        tmp3768 -= 1.0;
        tmp3767 += 1.0;
      }
      else if(tmp3768 < -0.5)
      {
        tmp3768 += 1.0;
        tmp3767 -= 1.0;
      }
      if(fabs(tmp3768) < 1e-10 && ((unsigned long)tmp3767 & 1))
      {
        tmp3764 = -pow(-tmp3762, tmp3765)*pow(tmp3762, tmp3766);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3762, tmp3763);
      }
    }
  }
  else
  {
    tmp3764 = pow(tmp3762, tmp3763);
  }
  if(isnan(tmp3764) || isinf(tmp3764))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3762, tmp3763);
  }tmp3769 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3764,"(r_init[124] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3769 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[124] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3769);
    }
  }
  (data->simulationInfo->realParameter[628] /* omega_c[124] PARAM */) = sqrt(tmp3769);
  TRACE_POP
}

/*
equation index: 13755
type: SIMPLE_ASSIGN
r_init[123] = r_min + 123.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13755};
  (data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (123.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13756
type: SIMPLE_ASSIGN
omega_c[123] = sqrt(G * Md / (r_init[123] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13756};
  modelica_real tmp3770;
  modelica_real tmp3771;
  modelica_real tmp3772;
  modelica_real tmp3773;
  modelica_real tmp3774;
  modelica_real tmp3775;
  modelica_real tmp3776;
  modelica_real tmp3777;
  modelica_real tmp3778;
  modelica_real tmp3779;
  tmp3770 = (data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */);
  tmp3771 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3772 = (tmp3770 * tmp3770) + (tmp3771 * tmp3771);
  tmp3773 = 1.5;
  if(tmp3772 < 0.0 && tmp3773 != 0.0)
  {
    tmp3775 = modf(tmp3773, &tmp3776);
    
    if(tmp3775 > 0.5)
    {
      tmp3775 -= 1.0;
      tmp3776 += 1.0;
    }
    else if(tmp3775 < -0.5)
    {
      tmp3775 += 1.0;
      tmp3776 -= 1.0;
    }
    
    if(fabs(tmp3775) < 1e-10)
      tmp3774 = pow(tmp3772, tmp3776);
    else
    {
      tmp3778 = modf(1.0/tmp3773, &tmp3777);
      if(tmp3778 > 0.5)
      {
        tmp3778 -= 1.0;
        tmp3777 += 1.0;
      }
      else if(tmp3778 < -0.5)
      {
        tmp3778 += 1.0;
        tmp3777 -= 1.0;
      }
      if(fabs(tmp3778) < 1e-10 && ((unsigned long)tmp3777 & 1))
      {
        tmp3774 = -pow(-tmp3772, tmp3775)*pow(tmp3772, tmp3776);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3772, tmp3773);
      }
    }
  }
  else
  {
    tmp3774 = pow(tmp3772, tmp3773);
  }
  if(isnan(tmp3774) || isinf(tmp3774))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3772, tmp3773);
  }tmp3779 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3774,"(r_init[123] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3779 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[123] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3779);
    }
  }
  (data->simulationInfo->realParameter[627] /* omega_c[123] PARAM */) = sqrt(tmp3779);
  TRACE_POP
}

/*
equation index: 13757
type: SIMPLE_ASSIGN
r_init[122] = r_min + 122.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13757};
  (data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (122.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13758
type: SIMPLE_ASSIGN
omega_c[122] = sqrt(G * Md / (r_init[122] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13758(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13758};
  modelica_real tmp3780;
  modelica_real tmp3781;
  modelica_real tmp3782;
  modelica_real tmp3783;
  modelica_real tmp3784;
  modelica_real tmp3785;
  modelica_real tmp3786;
  modelica_real tmp3787;
  modelica_real tmp3788;
  modelica_real tmp3789;
  tmp3780 = (data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */);
  tmp3781 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3782 = (tmp3780 * tmp3780) + (tmp3781 * tmp3781);
  tmp3783 = 1.5;
  if(tmp3782 < 0.0 && tmp3783 != 0.0)
  {
    tmp3785 = modf(tmp3783, &tmp3786);
    
    if(tmp3785 > 0.5)
    {
      tmp3785 -= 1.0;
      tmp3786 += 1.0;
    }
    else if(tmp3785 < -0.5)
    {
      tmp3785 += 1.0;
      tmp3786 -= 1.0;
    }
    
    if(fabs(tmp3785) < 1e-10)
      tmp3784 = pow(tmp3782, tmp3786);
    else
    {
      tmp3788 = modf(1.0/tmp3783, &tmp3787);
      if(tmp3788 > 0.5)
      {
        tmp3788 -= 1.0;
        tmp3787 += 1.0;
      }
      else if(tmp3788 < -0.5)
      {
        tmp3788 += 1.0;
        tmp3787 -= 1.0;
      }
      if(fabs(tmp3788) < 1e-10 && ((unsigned long)tmp3787 & 1))
      {
        tmp3784 = -pow(-tmp3782, tmp3785)*pow(tmp3782, tmp3786);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3782, tmp3783);
      }
    }
  }
  else
  {
    tmp3784 = pow(tmp3782, tmp3783);
  }
  if(isnan(tmp3784) || isinf(tmp3784))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3782, tmp3783);
  }tmp3789 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3784,"(r_init[122] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3789 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[122] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3789);
    }
  }
  (data->simulationInfo->realParameter[626] /* omega_c[122] PARAM */) = sqrt(tmp3789);
  TRACE_POP
}

/*
equation index: 13759
type: SIMPLE_ASSIGN
r_init[121] = r_min + 121.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13759};
  (data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (121.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13760
type: SIMPLE_ASSIGN
omega_c[121] = sqrt(G * Md / (r_init[121] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13760(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13760};
  modelica_real tmp3790;
  modelica_real tmp3791;
  modelica_real tmp3792;
  modelica_real tmp3793;
  modelica_real tmp3794;
  modelica_real tmp3795;
  modelica_real tmp3796;
  modelica_real tmp3797;
  modelica_real tmp3798;
  modelica_real tmp3799;
  tmp3790 = (data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */);
  tmp3791 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3792 = (tmp3790 * tmp3790) + (tmp3791 * tmp3791);
  tmp3793 = 1.5;
  if(tmp3792 < 0.0 && tmp3793 != 0.0)
  {
    tmp3795 = modf(tmp3793, &tmp3796);
    
    if(tmp3795 > 0.5)
    {
      tmp3795 -= 1.0;
      tmp3796 += 1.0;
    }
    else if(tmp3795 < -0.5)
    {
      tmp3795 += 1.0;
      tmp3796 -= 1.0;
    }
    
    if(fabs(tmp3795) < 1e-10)
      tmp3794 = pow(tmp3792, tmp3796);
    else
    {
      tmp3798 = modf(1.0/tmp3793, &tmp3797);
      if(tmp3798 > 0.5)
      {
        tmp3798 -= 1.0;
        tmp3797 += 1.0;
      }
      else if(tmp3798 < -0.5)
      {
        tmp3798 += 1.0;
        tmp3797 -= 1.0;
      }
      if(fabs(tmp3798) < 1e-10 && ((unsigned long)tmp3797 & 1))
      {
        tmp3794 = -pow(-tmp3792, tmp3795)*pow(tmp3792, tmp3796);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3792, tmp3793);
      }
    }
  }
  else
  {
    tmp3794 = pow(tmp3792, tmp3793);
  }
  if(isnan(tmp3794) || isinf(tmp3794))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3792, tmp3793);
  }tmp3799 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3794,"(r_init[121] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3799 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[121] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3799);
    }
  }
  (data->simulationInfo->realParameter[625] /* omega_c[121] PARAM */) = sqrt(tmp3799);
  TRACE_POP
}

/*
equation index: 13761
type: SIMPLE_ASSIGN
r_init[120] = r_min + 120.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13761};
  (data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (120.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13762
type: SIMPLE_ASSIGN
omega_c[120] = sqrt(G * Md / (r_init[120] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13762(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13762};
  modelica_real tmp3800;
  modelica_real tmp3801;
  modelica_real tmp3802;
  modelica_real tmp3803;
  modelica_real tmp3804;
  modelica_real tmp3805;
  modelica_real tmp3806;
  modelica_real tmp3807;
  modelica_real tmp3808;
  modelica_real tmp3809;
  tmp3800 = (data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */);
  tmp3801 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3802 = (tmp3800 * tmp3800) + (tmp3801 * tmp3801);
  tmp3803 = 1.5;
  if(tmp3802 < 0.0 && tmp3803 != 0.0)
  {
    tmp3805 = modf(tmp3803, &tmp3806);
    
    if(tmp3805 > 0.5)
    {
      tmp3805 -= 1.0;
      tmp3806 += 1.0;
    }
    else if(tmp3805 < -0.5)
    {
      tmp3805 += 1.0;
      tmp3806 -= 1.0;
    }
    
    if(fabs(tmp3805) < 1e-10)
      tmp3804 = pow(tmp3802, tmp3806);
    else
    {
      tmp3808 = modf(1.0/tmp3803, &tmp3807);
      if(tmp3808 > 0.5)
      {
        tmp3808 -= 1.0;
        tmp3807 += 1.0;
      }
      else if(tmp3808 < -0.5)
      {
        tmp3808 += 1.0;
        tmp3807 -= 1.0;
      }
      if(fabs(tmp3808) < 1e-10 && ((unsigned long)tmp3807 & 1))
      {
        tmp3804 = -pow(-tmp3802, tmp3805)*pow(tmp3802, tmp3806);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3802, tmp3803);
      }
    }
  }
  else
  {
    tmp3804 = pow(tmp3802, tmp3803);
  }
  if(isnan(tmp3804) || isinf(tmp3804))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3802, tmp3803);
  }tmp3809 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3804,"(r_init[120] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3809 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[120] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3809);
    }
  }
  (data->simulationInfo->realParameter[624] /* omega_c[120] PARAM */) = sqrt(tmp3809);
  TRACE_POP
}

/*
equation index: 13763
type: SIMPLE_ASSIGN
r_init[119] = r_min + 119.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13763};
  (data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (119.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13764
type: SIMPLE_ASSIGN
omega_c[119] = sqrt(G * Md / (r_init[119] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13764};
  modelica_real tmp3810;
  modelica_real tmp3811;
  modelica_real tmp3812;
  modelica_real tmp3813;
  modelica_real tmp3814;
  modelica_real tmp3815;
  modelica_real tmp3816;
  modelica_real tmp3817;
  modelica_real tmp3818;
  modelica_real tmp3819;
  tmp3810 = (data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */);
  tmp3811 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3812 = (tmp3810 * tmp3810) + (tmp3811 * tmp3811);
  tmp3813 = 1.5;
  if(tmp3812 < 0.0 && tmp3813 != 0.0)
  {
    tmp3815 = modf(tmp3813, &tmp3816);
    
    if(tmp3815 > 0.5)
    {
      tmp3815 -= 1.0;
      tmp3816 += 1.0;
    }
    else if(tmp3815 < -0.5)
    {
      tmp3815 += 1.0;
      tmp3816 -= 1.0;
    }
    
    if(fabs(tmp3815) < 1e-10)
      tmp3814 = pow(tmp3812, tmp3816);
    else
    {
      tmp3818 = modf(1.0/tmp3813, &tmp3817);
      if(tmp3818 > 0.5)
      {
        tmp3818 -= 1.0;
        tmp3817 += 1.0;
      }
      else if(tmp3818 < -0.5)
      {
        tmp3818 += 1.0;
        tmp3817 -= 1.0;
      }
      if(fabs(tmp3818) < 1e-10 && ((unsigned long)tmp3817 & 1))
      {
        tmp3814 = -pow(-tmp3812, tmp3815)*pow(tmp3812, tmp3816);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3812, tmp3813);
      }
    }
  }
  else
  {
    tmp3814 = pow(tmp3812, tmp3813);
  }
  if(isnan(tmp3814) || isinf(tmp3814))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3812, tmp3813);
  }tmp3819 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3814,"(r_init[119] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3819 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[119] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3819);
    }
  }
  (data->simulationInfo->realParameter[623] /* omega_c[119] PARAM */) = sqrt(tmp3819);
  TRACE_POP
}

/*
equation index: 13765
type: SIMPLE_ASSIGN
r_init[118] = r_min + 118.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13765};
  (data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (118.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13766
type: SIMPLE_ASSIGN
omega_c[118] = sqrt(G * Md / (r_init[118] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13766(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13766};
  modelica_real tmp3820;
  modelica_real tmp3821;
  modelica_real tmp3822;
  modelica_real tmp3823;
  modelica_real tmp3824;
  modelica_real tmp3825;
  modelica_real tmp3826;
  modelica_real tmp3827;
  modelica_real tmp3828;
  modelica_real tmp3829;
  tmp3820 = (data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */);
  tmp3821 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3822 = (tmp3820 * tmp3820) + (tmp3821 * tmp3821);
  tmp3823 = 1.5;
  if(tmp3822 < 0.0 && tmp3823 != 0.0)
  {
    tmp3825 = modf(tmp3823, &tmp3826);
    
    if(tmp3825 > 0.5)
    {
      tmp3825 -= 1.0;
      tmp3826 += 1.0;
    }
    else if(tmp3825 < -0.5)
    {
      tmp3825 += 1.0;
      tmp3826 -= 1.0;
    }
    
    if(fabs(tmp3825) < 1e-10)
      tmp3824 = pow(tmp3822, tmp3826);
    else
    {
      tmp3828 = modf(1.0/tmp3823, &tmp3827);
      if(tmp3828 > 0.5)
      {
        tmp3828 -= 1.0;
        tmp3827 += 1.0;
      }
      else if(tmp3828 < -0.5)
      {
        tmp3828 += 1.0;
        tmp3827 -= 1.0;
      }
      if(fabs(tmp3828) < 1e-10 && ((unsigned long)tmp3827 & 1))
      {
        tmp3824 = -pow(-tmp3822, tmp3825)*pow(tmp3822, tmp3826);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3822, tmp3823);
      }
    }
  }
  else
  {
    tmp3824 = pow(tmp3822, tmp3823);
  }
  if(isnan(tmp3824) || isinf(tmp3824))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3822, tmp3823);
  }tmp3829 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3824,"(r_init[118] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3829 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[118] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3829);
    }
  }
  (data->simulationInfo->realParameter[622] /* omega_c[118] PARAM */) = sqrt(tmp3829);
  TRACE_POP
}

/*
equation index: 13767
type: SIMPLE_ASSIGN
r_init[117] = r_min + 117.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13767};
  (data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (117.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13768
type: SIMPLE_ASSIGN
omega_c[117] = sqrt(G * Md / (r_init[117] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13768(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13768};
  modelica_real tmp3830;
  modelica_real tmp3831;
  modelica_real tmp3832;
  modelica_real tmp3833;
  modelica_real tmp3834;
  modelica_real tmp3835;
  modelica_real tmp3836;
  modelica_real tmp3837;
  modelica_real tmp3838;
  modelica_real tmp3839;
  tmp3830 = (data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */);
  tmp3831 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3832 = (tmp3830 * tmp3830) + (tmp3831 * tmp3831);
  tmp3833 = 1.5;
  if(tmp3832 < 0.0 && tmp3833 != 0.0)
  {
    tmp3835 = modf(tmp3833, &tmp3836);
    
    if(tmp3835 > 0.5)
    {
      tmp3835 -= 1.0;
      tmp3836 += 1.0;
    }
    else if(tmp3835 < -0.5)
    {
      tmp3835 += 1.0;
      tmp3836 -= 1.0;
    }
    
    if(fabs(tmp3835) < 1e-10)
      tmp3834 = pow(tmp3832, tmp3836);
    else
    {
      tmp3838 = modf(1.0/tmp3833, &tmp3837);
      if(tmp3838 > 0.5)
      {
        tmp3838 -= 1.0;
        tmp3837 += 1.0;
      }
      else if(tmp3838 < -0.5)
      {
        tmp3838 += 1.0;
        tmp3837 -= 1.0;
      }
      if(fabs(tmp3838) < 1e-10 && ((unsigned long)tmp3837 & 1))
      {
        tmp3834 = -pow(-tmp3832, tmp3835)*pow(tmp3832, tmp3836);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3832, tmp3833);
      }
    }
  }
  else
  {
    tmp3834 = pow(tmp3832, tmp3833);
  }
  if(isnan(tmp3834) || isinf(tmp3834))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3832, tmp3833);
  }tmp3839 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3834,"(r_init[117] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3839 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[117] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3839);
    }
  }
  (data->simulationInfo->realParameter[621] /* omega_c[117] PARAM */) = sqrt(tmp3839);
  TRACE_POP
}

/*
equation index: 13769
type: SIMPLE_ASSIGN
r_init[116] = r_min + 116.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13769};
  (data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (116.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13770
type: SIMPLE_ASSIGN
omega_c[116] = sqrt(G * Md / (r_init[116] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13770(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13770};
  modelica_real tmp3840;
  modelica_real tmp3841;
  modelica_real tmp3842;
  modelica_real tmp3843;
  modelica_real tmp3844;
  modelica_real tmp3845;
  modelica_real tmp3846;
  modelica_real tmp3847;
  modelica_real tmp3848;
  modelica_real tmp3849;
  tmp3840 = (data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */);
  tmp3841 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3842 = (tmp3840 * tmp3840) + (tmp3841 * tmp3841);
  tmp3843 = 1.5;
  if(tmp3842 < 0.0 && tmp3843 != 0.0)
  {
    tmp3845 = modf(tmp3843, &tmp3846);
    
    if(tmp3845 > 0.5)
    {
      tmp3845 -= 1.0;
      tmp3846 += 1.0;
    }
    else if(tmp3845 < -0.5)
    {
      tmp3845 += 1.0;
      tmp3846 -= 1.0;
    }
    
    if(fabs(tmp3845) < 1e-10)
      tmp3844 = pow(tmp3842, tmp3846);
    else
    {
      tmp3848 = modf(1.0/tmp3843, &tmp3847);
      if(tmp3848 > 0.5)
      {
        tmp3848 -= 1.0;
        tmp3847 += 1.0;
      }
      else if(tmp3848 < -0.5)
      {
        tmp3848 += 1.0;
        tmp3847 -= 1.0;
      }
      if(fabs(tmp3848) < 1e-10 && ((unsigned long)tmp3847 & 1))
      {
        tmp3844 = -pow(-tmp3842, tmp3845)*pow(tmp3842, tmp3846);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3842, tmp3843);
      }
    }
  }
  else
  {
    tmp3844 = pow(tmp3842, tmp3843);
  }
  if(isnan(tmp3844) || isinf(tmp3844))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3842, tmp3843);
  }tmp3849 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3844,"(r_init[116] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3849 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[116] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3849);
    }
  }
  (data->simulationInfo->realParameter[620] /* omega_c[116] PARAM */) = sqrt(tmp3849);
  TRACE_POP
}

/*
equation index: 13771
type: SIMPLE_ASSIGN
r_init[115] = r_min + 115.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13771};
  (data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (115.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13772
type: SIMPLE_ASSIGN
omega_c[115] = sqrt(G * Md / (r_init[115] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13772};
  modelica_real tmp3850;
  modelica_real tmp3851;
  modelica_real tmp3852;
  modelica_real tmp3853;
  modelica_real tmp3854;
  modelica_real tmp3855;
  modelica_real tmp3856;
  modelica_real tmp3857;
  modelica_real tmp3858;
  modelica_real tmp3859;
  tmp3850 = (data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */);
  tmp3851 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3852 = (tmp3850 * tmp3850) + (tmp3851 * tmp3851);
  tmp3853 = 1.5;
  if(tmp3852 < 0.0 && tmp3853 != 0.0)
  {
    tmp3855 = modf(tmp3853, &tmp3856);
    
    if(tmp3855 > 0.5)
    {
      tmp3855 -= 1.0;
      tmp3856 += 1.0;
    }
    else if(tmp3855 < -0.5)
    {
      tmp3855 += 1.0;
      tmp3856 -= 1.0;
    }
    
    if(fabs(tmp3855) < 1e-10)
      tmp3854 = pow(tmp3852, tmp3856);
    else
    {
      tmp3858 = modf(1.0/tmp3853, &tmp3857);
      if(tmp3858 > 0.5)
      {
        tmp3858 -= 1.0;
        tmp3857 += 1.0;
      }
      else if(tmp3858 < -0.5)
      {
        tmp3858 += 1.0;
        tmp3857 -= 1.0;
      }
      if(fabs(tmp3858) < 1e-10 && ((unsigned long)tmp3857 & 1))
      {
        tmp3854 = -pow(-tmp3852, tmp3855)*pow(tmp3852, tmp3856);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3852, tmp3853);
      }
    }
  }
  else
  {
    tmp3854 = pow(tmp3852, tmp3853);
  }
  if(isnan(tmp3854) || isinf(tmp3854))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3852, tmp3853);
  }tmp3859 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3854,"(r_init[115] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3859 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[115] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3859);
    }
  }
  (data->simulationInfo->realParameter[619] /* omega_c[115] PARAM */) = sqrt(tmp3859);
  TRACE_POP
}

/*
equation index: 13773
type: SIMPLE_ASSIGN
r_init[114] = r_min + 114.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13773};
  (data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (114.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13774
type: SIMPLE_ASSIGN
omega_c[114] = sqrt(G * Md / (r_init[114] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13774(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13774};
  modelica_real tmp3860;
  modelica_real tmp3861;
  modelica_real tmp3862;
  modelica_real tmp3863;
  modelica_real tmp3864;
  modelica_real tmp3865;
  modelica_real tmp3866;
  modelica_real tmp3867;
  modelica_real tmp3868;
  modelica_real tmp3869;
  tmp3860 = (data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */);
  tmp3861 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3862 = (tmp3860 * tmp3860) + (tmp3861 * tmp3861);
  tmp3863 = 1.5;
  if(tmp3862 < 0.0 && tmp3863 != 0.0)
  {
    tmp3865 = modf(tmp3863, &tmp3866);
    
    if(tmp3865 > 0.5)
    {
      tmp3865 -= 1.0;
      tmp3866 += 1.0;
    }
    else if(tmp3865 < -0.5)
    {
      tmp3865 += 1.0;
      tmp3866 -= 1.0;
    }
    
    if(fabs(tmp3865) < 1e-10)
      tmp3864 = pow(tmp3862, tmp3866);
    else
    {
      tmp3868 = modf(1.0/tmp3863, &tmp3867);
      if(tmp3868 > 0.5)
      {
        tmp3868 -= 1.0;
        tmp3867 += 1.0;
      }
      else if(tmp3868 < -0.5)
      {
        tmp3868 += 1.0;
        tmp3867 -= 1.0;
      }
      if(fabs(tmp3868) < 1e-10 && ((unsigned long)tmp3867 & 1))
      {
        tmp3864 = -pow(-tmp3862, tmp3865)*pow(tmp3862, tmp3866);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3862, tmp3863);
      }
    }
  }
  else
  {
    tmp3864 = pow(tmp3862, tmp3863);
  }
  if(isnan(tmp3864) || isinf(tmp3864))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3862, tmp3863);
  }tmp3869 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3864,"(r_init[114] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3869 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[114] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3869);
    }
  }
  (data->simulationInfo->realParameter[618] /* omega_c[114] PARAM */) = sqrt(tmp3869);
  TRACE_POP
}

/*
equation index: 13775
type: SIMPLE_ASSIGN
r_init[113] = r_min + 113.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13775};
  (data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (113.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13776
type: SIMPLE_ASSIGN
omega_c[113] = sqrt(G * Md / (r_init[113] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13776(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13776};
  modelica_real tmp3870;
  modelica_real tmp3871;
  modelica_real tmp3872;
  modelica_real tmp3873;
  modelica_real tmp3874;
  modelica_real tmp3875;
  modelica_real tmp3876;
  modelica_real tmp3877;
  modelica_real tmp3878;
  modelica_real tmp3879;
  tmp3870 = (data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */);
  tmp3871 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3872 = (tmp3870 * tmp3870) + (tmp3871 * tmp3871);
  tmp3873 = 1.5;
  if(tmp3872 < 0.0 && tmp3873 != 0.0)
  {
    tmp3875 = modf(tmp3873, &tmp3876);
    
    if(tmp3875 > 0.5)
    {
      tmp3875 -= 1.0;
      tmp3876 += 1.0;
    }
    else if(tmp3875 < -0.5)
    {
      tmp3875 += 1.0;
      tmp3876 -= 1.0;
    }
    
    if(fabs(tmp3875) < 1e-10)
      tmp3874 = pow(tmp3872, tmp3876);
    else
    {
      tmp3878 = modf(1.0/tmp3873, &tmp3877);
      if(tmp3878 > 0.5)
      {
        tmp3878 -= 1.0;
        tmp3877 += 1.0;
      }
      else if(tmp3878 < -0.5)
      {
        tmp3878 += 1.0;
        tmp3877 -= 1.0;
      }
      if(fabs(tmp3878) < 1e-10 && ((unsigned long)tmp3877 & 1))
      {
        tmp3874 = -pow(-tmp3872, tmp3875)*pow(tmp3872, tmp3876);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3872, tmp3873);
      }
    }
  }
  else
  {
    tmp3874 = pow(tmp3872, tmp3873);
  }
  if(isnan(tmp3874) || isinf(tmp3874))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3872, tmp3873);
  }tmp3879 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3874,"(r_init[113] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3879 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[113] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3879);
    }
  }
  (data->simulationInfo->realParameter[617] /* omega_c[113] PARAM */) = sqrt(tmp3879);
  TRACE_POP
}

/*
equation index: 13777
type: SIMPLE_ASSIGN
r_init[112] = r_min + 112.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13777};
  (data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (112.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13778
type: SIMPLE_ASSIGN
omega_c[112] = sqrt(G * Md / (r_init[112] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13778(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13778};
  modelica_real tmp3880;
  modelica_real tmp3881;
  modelica_real tmp3882;
  modelica_real tmp3883;
  modelica_real tmp3884;
  modelica_real tmp3885;
  modelica_real tmp3886;
  modelica_real tmp3887;
  modelica_real tmp3888;
  modelica_real tmp3889;
  tmp3880 = (data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */);
  tmp3881 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3882 = (tmp3880 * tmp3880) + (tmp3881 * tmp3881);
  tmp3883 = 1.5;
  if(tmp3882 < 0.0 && tmp3883 != 0.0)
  {
    tmp3885 = modf(tmp3883, &tmp3886);
    
    if(tmp3885 > 0.5)
    {
      tmp3885 -= 1.0;
      tmp3886 += 1.0;
    }
    else if(tmp3885 < -0.5)
    {
      tmp3885 += 1.0;
      tmp3886 -= 1.0;
    }
    
    if(fabs(tmp3885) < 1e-10)
      tmp3884 = pow(tmp3882, tmp3886);
    else
    {
      tmp3888 = modf(1.0/tmp3883, &tmp3887);
      if(tmp3888 > 0.5)
      {
        tmp3888 -= 1.0;
        tmp3887 += 1.0;
      }
      else if(tmp3888 < -0.5)
      {
        tmp3888 += 1.0;
        tmp3887 -= 1.0;
      }
      if(fabs(tmp3888) < 1e-10 && ((unsigned long)tmp3887 & 1))
      {
        tmp3884 = -pow(-tmp3882, tmp3885)*pow(tmp3882, tmp3886);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3882, tmp3883);
      }
    }
  }
  else
  {
    tmp3884 = pow(tmp3882, tmp3883);
  }
  if(isnan(tmp3884) || isinf(tmp3884))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3882, tmp3883);
  }tmp3889 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3884,"(r_init[112] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3889 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[112] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3889);
    }
  }
  (data->simulationInfo->realParameter[616] /* omega_c[112] PARAM */) = sqrt(tmp3889);
  TRACE_POP
}

/*
equation index: 13779
type: SIMPLE_ASSIGN
r_init[111] = r_min + 111.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13779};
  (data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (111.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13780
type: SIMPLE_ASSIGN
omega_c[111] = sqrt(G * Md / (r_init[111] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13780};
  modelica_real tmp3890;
  modelica_real tmp3891;
  modelica_real tmp3892;
  modelica_real tmp3893;
  modelica_real tmp3894;
  modelica_real tmp3895;
  modelica_real tmp3896;
  modelica_real tmp3897;
  modelica_real tmp3898;
  modelica_real tmp3899;
  tmp3890 = (data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */);
  tmp3891 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3892 = (tmp3890 * tmp3890) + (tmp3891 * tmp3891);
  tmp3893 = 1.5;
  if(tmp3892 < 0.0 && tmp3893 != 0.0)
  {
    tmp3895 = modf(tmp3893, &tmp3896);
    
    if(tmp3895 > 0.5)
    {
      tmp3895 -= 1.0;
      tmp3896 += 1.0;
    }
    else if(tmp3895 < -0.5)
    {
      tmp3895 += 1.0;
      tmp3896 -= 1.0;
    }
    
    if(fabs(tmp3895) < 1e-10)
      tmp3894 = pow(tmp3892, tmp3896);
    else
    {
      tmp3898 = modf(1.0/tmp3893, &tmp3897);
      if(tmp3898 > 0.5)
      {
        tmp3898 -= 1.0;
        tmp3897 += 1.0;
      }
      else if(tmp3898 < -0.5)
      {
        tmp3898 += 1.0;
        tmp3897 -= 1.0;
      }
      if(fabs(tmp3898) < 1e-10 && ((unsigned long)tmp3897 & 1))
      {
        tmp3894 = -pow(-tmp3892, tmp3895)*pow(tmp3892, tmp3896);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3892, tmp3893);
      }
    }
  }
  else
  {
    tmp3894 = pow(tmp3892, tmp3893);
  }
  if(isnan(tmp3894) || isinf(tmp3894))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3892, tmp3893);
  }tmp3899 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3894,"(r_init[111] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3899 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[111] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3899);
    }
  }
  (data->simulationInfo->realParameter[615] /* omega_c[111] PARAM */) = sqrt(tmp3899);
  TRACE_POP
}

/*
equation index: 13781
type: SIMPLE_ASSIGN
r_init[110] = r_min + 110.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13781};
  (data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (110.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13782
type: SIMPLE_ASSIGN
omega_c[110] = sqrt(G * Md / (r_init[110] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13782(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13782};
  modelica_real tmp3900;
  modelica_real tmp3901;
  modelica_real tmp3902;
  modelica_real tmp3903;
  modelica_real tmp3904;
  modelica_real tmp3905;
  modelica_real tmp3906;
  modelica_real tmp3907;
  modelica_real tmp3908;
  modelica_real tmp3909;
  tmp3900 = (data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */);
  tmp3901 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3902 = (tmp3900 * tmp3900) + (tmp3901 * tmp3901);
  tmp3903 = 1.5;
  if(tmp3902 < 0.0 && tmp3903 != 0.0)
  {
    tmp3905 = modf(tmp3903, &tmp3906);
    
    if(tmp3905 > 0.5)
    {
      tmp3905 -= 1.0;
      tmp3906 += 1.0;
    }
    else if(tmp3905 < -0.5)
    {
      tmp3905 += 1.0;
      tmp3906 -= 1.0;
    }
    
    if(fabs(tmp3905) < 1e-10)
      tmp3904 = pow(tmp3902, tmp3906);
    else
    {
      tmp3908 = modf(1.0/tmp3903, &tmp3907);
      if(tmp3908 > 0.5)
      {
        tmp3908 -= 1.0;
        tmp3907 += 1.0;
      }
      else if(tmp3908 < -0.5)
      {
        tmp3908 += 1.0;
        tmp3907 -= 1.0;
      }
      if(fabs(tmp3908) < 1e-10 && ((unsigned long)tmp3907 & 1))
      {
        tmp3904 = -pow(-tmp3902, tmp3905)*pow(tmp3902, tmp3906);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3902, tmp3903);
      }
    }
  }
  else
  {
    tmp3904 = pow(tmp3902, tmp3903);
  }
  if(isnan(tmp3904) || isinf(tmp3904))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3902, tmp3903);
  }tmp3909 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3904,"(r_init[110] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3909 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[110] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3909);
    }
  }
  (data->simulationInfo->realParameter[614] /* omega_c[110] PARAM */) = sqrt(tmp3909);
  TRACE_POP
}

/*
equation index: 13783
type: SIMPLE_ASSIGN
r_init[109] = r_min + 109.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13783};
  (data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (109.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13784
type: SIMPLE_ASSIGN
omega_c[109] = sqrt(G * Md / (r_init[109] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13784(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13784};
  modelica_real tmp3910;
  modelica_real tmp3911;
  modelica_real tmp3912;
  modelica_real tmp3913;
  modelica_real tmp3914;
  modelica_real tmp3915;
  modelica_real tmp3916;
  modelica_real tmp3917;
  modelica_real tmp3918;
  modelica_real tmp3919;
  tmp3910 = (data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */);
  tmp3911 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3912 = (tmp3910 * tmp3910) + (tmp3911 * tmp3911);
  tmp3913 = 1.5;
  if(tmp3912 < 0.0 && tmp3913 != 0.0)
  {
    tmp3915 = modf(tmp3913, &tmp3916);
    
    if(tmp3915 > 0.5)
    {
      tmp3915 -= 1.0;
      tmp3916 += 1.0;
    }
    else if(tmp3915 < -0.5)
    {
      tmp3915 += 1.0;
      tmp3916 -= 1.0;
    }
    
    if(fabs(tmp3915) < 1e-10)
      tmp3914 = pow(tmp3912, tmp3916);
    else
    {
      tmp3918 = modf(1.0/tmp3913, &tmp3917);
      if(tmp3918 > 0.5)
      {
        tmp3918 -= 1.0;
        tmp3917 += 1.0;
      }
      else if(tmp3918 < -0.5)
      {
        tmp3918 += 1.0;
        tmp3917 -= 1.0;
      }
      if(fabs(tmp3918) < 1e-10 && ((unsigned long)tmp3917 & 1))
      {
        tmp3914 = -pow(-tmp3912, tmp3915)*pow(tmp3912, tmp3916);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3912, tmp3913);
      }
    }
  }
  else
  {
    tmp3914 = pow(tmp3912, tmp3913);
  }
  if(isnan(tmp3914) || isinf(tmp3914))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3912, tmp3913);
  }tmp3919 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3914,"(r_init[109] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3919 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[109] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3919);
    }
  }
  (data->simulationInfo->realParameter[613] /* omega_c[109] PARAM */) = sqrt(tmp3919);
  TRACE_POP
}

/*
equation index: 13785
type: SIMPLE_ASSIGN
r_init[108] = r_min + 108.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13785};
  (data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (108.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13786
type: SIMPLE_ASSIGN
omega_c[108] = sqrt(G * Md / (r_init[108] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13786(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13786};
  modelica_real tmp3920;
  modelica_real tmp3921;
  modelica_real tmp3922;
  modelica_real tmp3923;
  modelica_real tmp3924;
  modelica_real tmp3925;
  modelica_real tmp3926;
  modelica_real tmp3927;
  modelica_real tmp3928;
  modelica_real tmp3929;
  tmp3920 = (data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */);
  tmp3921 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3922 = (tmp3920 * tmp3920) + (tmp3921 * tmp3921);
  tmp3923 = 1.5;
  if(tmp3922 < 0.0 && tmp3923 != 0.0)
  {
    tmp3925 = modf(tmp3923, &tmp3926);
    
    if(tmp3925 > 0.5)
    {
      tmp3925 -= 1.0;
      tmp3926 += 1.0;
    }
    else if(tmp3925 < -0.5)
    {
      tmp3925 += 1.0;
      tmp3926 -= 1.0;
    }
    
    if(fabs(tmp3925) < 1e-10)
      tmp3924 = pow(tmp3922, tmp3926);
    else
    {
      tmp3928 = modf(1.0/tmp3923, &tmp3927);
      if(tmp3928 > 0.5)
      {
        tmp3928 -= 1.0;
        tmp3927 += 1.0;
      }
      else if(tmp3928 < -0.5)
      {
        tmp3928 += 1.0;
        tmp3927 -= 1.0;
      }
      if(fabs(tmp3928) < 1e-10 && ((unsigned long)tmp3927 & 1))
      {
        tmp3924 = -pow(-tmp3922, tmp3925)*pow(tmp3922, tmp3926);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3922, tmp3923);
      }
    }
  }
  else
  {
    tmp3924 = pow(tmp3922, tmp3923);
  }
  if(isnan(tmp3924) || isinf(tmp3924))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3922, tmp3923);
  }tmp3929 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3924,"(r_init[108] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3929 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[108] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3929);
    }
  }
  (data->simulationInfo->realParameter[612] /* omega_c[108] PARAM */) = sqrt(tmp3929);
  TRACE_POP
}

/*
equation index: 13787
type: SIMPLE_ASSIGN
r_init[107] = r_min + 107.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13787};
  (data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (107.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13788
type: SIMPLE_ASSIGN
omega_c[107] = sqrt(G * Md / (r_init[107] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13788};
  modelica_real tmp3930;
  modelica_real tmp3931;
  modelica_real tmp3932;
  modelica_real tmp3933;
  modelica_real tmp3934;
  modelica_real tmp3935;
  modelica_real tmp3936;
  modelica_real tmp3937;
  modelica_real tmp3938;
  modelica_real tmp3939;
  tmp3930 = (data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */);
  tmp3931 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3932 = (tmp3930 * tmp3930) + (tmp3931 * tmp3931);
  tmp3933 = 1.5;
  if(tmp3932 < 0.0 && tmp3933 != 0.0)
  {
    tmp3935 = modf(tmp3933, &tmp3936);
    
    if(tmp3935 > 0.5)
    {
      tmp3935 -= 1.0;
      tmp3936 += 1.0;
    }
    else if(tmp3935 < -0.5)
    {
      tmp3935 += 1.0;
      tmp3936 -= 1.0;
    }
    
    if(fabs(tmp3935) < 1e-10)
      tmp3934 = pow(tmp3932, tmp3936);
    else
    {
      tmp3938 = modf(1.0/tmp3933, &tmp3937);
      if(tmp3938 > 0.5)
      {
        tmp3938 -= 1.0;
        tmp3937 += 1.0;
      }
      else if(tmp3938 < -0.5)
      {
        tmp3938 += 1.0;
        tmp3937 -= 1.0;
      }
      if(fabs(tmp3938) < 1e-10 && ((unsigned long)tmp3937 & 1))
      {
        tmp3934 = -pow(-tmp3932, tmp3935)*pow(tmp3932, tmp3936);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3932, tmp3933);
      }
    }
  }
  else
  {
    tmp3934 = pow(tmp3932, tmp3933);
  }
  if(isnan(tmp3934) || isinf(tmp3934))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3932, tmp3933);
  }tmp3939 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3934,"(r_init[107] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3939 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[107] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3939);
    }
  }
  (data->simulationInfo->realParameter[611] /* omega_c[107] PARAM */) = sqrt(tmp3939);
  TRACE_POP
}

/*
equation index: 13789
type: SIMPLE_ASSIGN
r_init[106] = r_min + 106.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13789};
  (data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (106.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13790
type: SIMPLE_ASSIGN
omega_c[106] = sqrt(G * Md / (r_init[106] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13790(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13790};
  modelica_real tmp3940;
  modelica_real tmp3941;
  modelica_real tmp3942;
  modelica_real tmp3943;
  modelica_real tmp3944;
  modelica_real tmp3945;
  modelica_real tmp3946;
  modelica_real tmp3947;
  modelica_real tmp3948;
  modelica_real tmp3949;
  tmp3940 = (data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */);
  tmp3941 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3942 = (tmp3940 * tmp3940) + (tmp3941 * tmp3941);
  tmp3943 = 1.5;
  if(tmp3942 < 0.0 && tmp3943 != 0.0)
  {
    tmp3945 = modf(tmp3943, &tmp3946);
    
    if(tmp3945 > 0.5)
    {
      tmp3945 -= 1.0;
      tmp3946 += 1.0;
    }
    else if(tmp3945 < -0.5)
    {
      tmp3945 += 1.0;
      tmp3946 -= 1.0;
    }
    
    if(fabs(tmp3945) < 1e-10)
      tmp3944 = pow(tmp3942, tmp3946);
    else
    {
      tmp3948 = modf(1.0/tmp3943, &tmp3947);
      if(tmp3948 > 0.5)
      {
        tmp3948 -= 1.0;
        tmp3947 += 1.0;
      }
      else if(tmp3948 < -0.5)
      {
        tmp3948 += 1.0;
        tmp3947 -= 1.0;
      }
      if(fabs(tmp3948) < 1e-10 && ((unsigned long)tmp3947 & 1))
      {
        tmp3944 = -pow(-tmp3942, tmp3945)*pow(tmp3942, tmp3946);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3942, tmp3943);
      }
    }
  }
  else
  {
    tmp3944 = pow(tmp3942, tmp3943);
  }
  if(isnan(tmp3944) || isinf(tmp3944))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3942, tmp3943);
  }tmp3949 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3944,"(r_init[106] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3949 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[106] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3949);
    }
  }
  (data->simulationInfo->realParameter[610] /* omega_c[106] PARAM */) = sqrt(tmp3949);
  TRACE_POP
}

/*
equation index: 13791
type: SIMPLE_ASSIGN
r_init[105] = r_min + 105.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13791};
  (data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (105.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13792
type: SIMPLE_ASSIGN
omega_c[105] = sqrt(G * Md / (r_init[105] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13792(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13792};
  modelica_real tmp3950;
  modelica_real tmp3951;
  modelica_real tmp3952;
  modelica_real tmp3953;
  modelica_real tmp3954;
  modelica_real tmp3955;
  modelica_real tmp3956;
  modelica_real tmp3957;
  modelica_real tmp3958;
  modelica_real tmp3959;
  tmp3950 = (data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */);
  tmp3951 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3952 = (tmp3950 * tmp3950) + (tmp3951 * tmp3951);
  tmp3953 = 1.5;
  if(tmp3952 < 0.0 && tmp3953 != 0.0)
  {
    tmp3955 = modf(tmp3953, &tmp3956);
    
    if(tmp3955 > 0.5)
    {
      tmp3955 -= 1.0;
      tmp3956 += 1.0;
    }
    else if(tmp3955 < -0.5)
    {
      tmp3955 += 1.0;
      tmp3956 -= 1.0;
    }
    
    if(fabs(tmp3955) < 1e-10)
      tmp3954 = pow(tmp3952, tmp3956);
    else
    {
      tmp3958 = modf(1.0/tmp3953, &tmp3957);
      if(tmp3958 > 0.5)
      {
        tmp3958 -= 1.0;
        tmp3957 += 1.0;
      }
      else if(tmp3958 < -0.5)
      {
        tmp3958 += 1.0;
        tmp3957 -= 1.0;
      }
      if(fabs(tmp3958) < 1e-10 && ((unsigned long)tmp3957 & 1))
      {
        tmp3954 = -pow(-tmp3952, tmp3955)*pow(tmp3952, tmp3956);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3952, tmp3953);
      }
    }
  }
  else
  {
    tmp3954 = pow(tmp3952, tmp3953);
  }
  if(isnan(tmp3954) || isinf(tmp3954))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3952, tmp3953);
  }tmp3959 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3954,"(r_init[105] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3959 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[105] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3959);
    }
  }
  (data->simulationInfo->realParameter[609] /* omega_c[105] PARAM */) = sqrt(tmp3959);
  TRACE_POP
}

/*
equation index: 13793
type: SIMPLE_ASSIGN
r_init[104] = r_min + 104.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13793};
  (data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (104.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13794
type: SIMPLE_ASSIGN
omega_c[104] = sqrt(G * Md / (r_init[104] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13794(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13794};
  modelica_real tmp3960;
  modelica_real tmp3961;
  modelica_real tmp3962;
  modelica_real tmp3963;
  modelica_real tmp3964;
  modelica_real tmp3965;
  modelica_real tmp3966;
  modelica_real tmp3967;
  modelica_real tmp3968;
  modelica_real tmp3969;
  tmp3960 = (data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */);
  tmp3961 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3962 = (tmp3960 * tmp3960) + (tmp3961 * tmp3961);
  tmp3963 = 1.5;
  if(tmp3962 < 0.0 && tmp3963 != 0.0)
  {
    tmp3965 = modf(tmp3963, &tmp3966);
    
    if(tmp3965 > 0.5)
    {
      tmp3965 -= 1.0;
      tmp3966 += 1.0;
    }
    else if(tmp3965 < -0.5)
    {
      tmp3965 += 1.0;
      tmp3966 -= 1.0;
    }
    
    if(fabs(tmp3965) < 1e-10)
      tmp3964 = pow(tmp3962, tmp3966);
    else
    {
      tmp3968 = modf(1.0/tmp3963, &tmp3967);
      if(tmp3968 > 0.5)
      {
        tmp3968 -= 1.0;
        tmp3967 += 1.0;
      }
      else if(tmp3968 < -0.5)
      {
        tmp3968 += 1.0;
        tmp3967 -= 1.0;
      }
      if(fabs(tmp3968) < 1e-10 && ((unsigned long)tmp3967 & 1))
      {
        tmp3964 = -pow(-tmp3962, tmp3965)*pow(tmp3962, tmp3966);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3962, tmp3963);
      }
    }
  }
  else
  {
    tmp3964 = pow(tmp3962, tmp3963);
  }
  if(isnan(tmp3964) || isinf(tmp3964))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3962, tmp3963);
  }tmp3969 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3964,"(r_init[104] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3969 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[104] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3969);
    }
  }
  (data->simulationInfo->realParameter[608] /* omega_c[104] PARAM */) = sqrt(tmp3969);
  TRACE_POP
}

/*
equation index: 13795
type: SIMPLE_ASSIGN
r_init[103] = r_min + 103.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13795};
  (data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (103.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13796
type: SIMPLE_ASSIGN
omega_c[103] = sqrt(G * Md / (r_init[103] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13796};
  modelica_real tmp3970;
  modelica_real tmp3971;
  modelica_real tmp3972;
  modelica_real tmp3973;
  modelica_real tmp3974;
  modelica_real tmp3975;
  modelica_real tmp3976;
  modelica_real tmp3977;
  modelica_real tmp3978;
  modelica_real tmp3979;
  tmp3970 = (data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */);
  tmp3971 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3972 = (tmp3970 * tmp3970) + (tmp3971 * tmp3971);
  tmp3973 = 1.5;
  if(tmp3972 < 0.0 && tmp3973 != 0.0)
  {
    tmp3975 = modf(tmp3973, &tmp3976);
    
    if(tmp3975 > 0.5)
    {
      tmp3975 -= 1.0;
      tmp3976 += 1.0;
    }
    else if(tmp3975 < -0.5)
    {
      tmp3975 += 1.0;
      tmp3976 -= 1.0;
    }
    
    if(fabs(tmp3975) < 1e-10)
      tmp3974 = pow(tmp3972, tmp3976);
    else
    {
      tmp3978 = modf(1.0/tmp3973, &tmp3977);
      if(tmp3978 > 0.5)
      {
        tmp3978 -= 1.0;
        tmp3977 += 1.0;
      }
      else if(tmp3978 < -0.5)
      {
        tmp3978 += 1.0;
        tmp3977 -= 1.0;
      }
      if(fabs(tmp3978) < 1e-10 && ((unsigned long)tmp3977 & 1))
      {
        tmp3974 = -pow(-tmp3972, tmp3975)*pow(tmp3972, tmp3976);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3972, tmp3973);
      }
    }
  }
  else
  {
    tmp3974 = pow(tmp3972, tmp3973);
  }
  if(isnan(tmp3974) || isinf(tmp3974))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3972, tmp3973);
  }tmp3979 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3974,"(r_init[103] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3979 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[103] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3979);
    }
  }
  (data->simulationInfo->realParameter[607] /* omega_c[103] PARAM */) = sqrt(tmp3979);
  TRACE_POP
}

/*
equation index: 13797
type: SIMPLE_ASSIGN
r_init[102] = r_min + 102.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13797};
  (data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (102.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13798
type: SIMPLE_ASSIGN
omega_c[102] = sqrt(G * Md / (r_init[102] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13798(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13798};
  modelica_real tmp3980;
  modelica_real tmp3981;
  modelica_real tmp3982;
  modelica_real tmp3983;
  modelica_real tmp3984;
  modelica_real tmp3985;
  modelica_real tmp3986;
  modelica_real tmp3987;
  modelica_real tmp3988;
  modelica_real tmp3989;
  tmp3980 = (data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */);
  tmp3981 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3982 = (tmp3980 * tmp3980) + (tmp3981 * tmp3981);
  tmp3983 = 1.5;
  if(tmp3982 < 0.0 && tmp3983 != 0.0)
  {
    tmp3985 = modf(tmp3983, &tmp3986);
    
    if(tmp3985 > 0.5)
    {
      tmp3985 -= 1.0;
      tmp3986 += 1.0;
    }
    else if(tmp3985 < -0.5)
    {
      tmp3985 += 1.0;
      tmp3986 -= 1.0;
    }
    
    if(fabs(tmp3985) < 1e-10)
      tmp3984 = pow(tmp3982, tmp3986);
    else
    {
      tmp3988 = modf(1.0/tmp3983, &tmp3987);
      if(tmp3988 > 0.5)
      {
        tmp3988 -= 1.0;
        tmp3987 += 1.0;
      }
      else if(tmp3988 < -0.5)
      {
        tmp3988 += 1.0;
        tmp3987 -= 1.0;
      }
      if(fabs(tmp3988) < 1e-10 && ((unsigned long)tmp3987 & 1))
      {
        tmp3984 = -pow(-tmp3982, tmp3985)*pow(tmp3982, tmp3986);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3982, tmp3983);
      }
    }
  }
  else
  {
    tmp3984 = pow(tmp3982, tmp3983);
  }
  if(isnan(tmp3984) || isinf(tmp3984))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3982, tmp3983);
  }tmp3989 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3984,"(r_init[102] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3989 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[102] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3989);
    }
  }
  (data->simulationInfo->realParameter[606] /* omega_c[102] PARAM */) = sqrt(tmp3989);
  TRACE_POP
}

/*
equation index: 13799
type: SIMPLE_ASSIGN
r_init[101] = r_min + 101.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13799};
  (data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (101.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13800
type: SIMPLE_ASSIGN
omega_c[101] = sqrt(G * Md / (r_init[101] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13800(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13800};
  modelica_real tmp3990;
  modelica_real tmp3991;
  modelica_real tmp3992;
  modelica_real tmp3993;
  modelica_real tmp3994;
  modelica_real tmp3995;
  modelica_real tmp3996;
  modelica_real tmp3997;
  modelica_real tmp3998;
  modelica_real tmp3999;
  tmp3990 = (data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */);
  tmp3991 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp3992 = (tmp3990 * tmp3990) + (tmp3991 * tmp3991);
  tmp3993 = 1.5;
  if(tmp3992 < 0.0 && tmp3993 != 0.0)
  {
    tmp3995 = modf(tmp3993, &tmp3996);
    
    if(tmp3995 > 0.5)
    {
      tmp3995 -= 1.0;
      tmp3996 += 1.0;
    }
    else if(tmp3995 < -0.5)
    {
      tmp3995 += 1.0;
      tmp3996 -= 1.0;
    }
    
    if(fabs(tmp3995) < 1e-10)
      tmp3994 = pow(tmp3992, tmp3996);
    else
    {
      tmp3998 = modf(1.0/tmp3993, &tmp3997);
      if(tmp3998 > 0.5)
      {
        tmp3998 -= 1.0;
        tmp3997 += 1.0;
      }
      else if(tmp3998 < -0.5)
      {
        tmp3998 += 1.0;
        tmp3997 -= 1.0;
      }
      if(fabs(tmp3998) < 1e-10 && ((unsigned long)tmp3997 & 1))
      {
        tmp3994 = -pow(-tmp3992, tmp3995)*pow(tmp3992, tmp3996);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3992, tmp3993);
      }
    }
  }
  else
  {
    tmp3994 = pow(tmp3992, tmp3993);
  }
  if(isnan(tmp3994) || isinf(tmp3994))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3992, tmp3993);
  }tmp3999 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp3994,"(r_init[101] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp3999 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[101] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp3999);
    }
  }
  (data->simulationInfo->realParameter[605] /* omega_c[101] PARAM */) = sqrt(tmp3999);
  TRACE_POP
}

/*
equation index: 13801
type: SIMPLE_ASSIGN
r_init[100] = r_min + 100.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13801};
  (data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (100.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13802
type: SIMPLE_ASSIGN
omega_c[100] = sqrt(G * Md / (r_init[100] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13802(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13802};
  modelica_real tmp4000;
  modelica_real tmp4001;
  modelica_real tmp4002;
  modelica_real tmp4003;
  modelica_real tmp4004;
  modelica_real tmp4005;
  modelica_real tmp4006;
  modelica_real tmp4007;
  modelica_real tmp4008;
  modelica_real tmp4009;
  tmp4000 = (data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */);
  tmp4001 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4002 = (tmp4000 * tmp4000) + (tmp4001 * tmp4001);
  tmp4003 = 1.5;
  if(tmp4002 < 0.0 && tmp4003 != 0.0)
  {
    tmp4005 = modf(tmp4003, &tmp4006);
    
    if(tmp4005 > 0.5)
    {
      tmp4005 -= 1.0;
      tmp4006 += 1.0;
    }
    else if(tmp4005 < -0.5)
    {
      tmp4005 += 1.0;
      tmp4006 -= 1.0;
    }
    
    if(fabs(tmp4005) < 1e-10)
      tmp4004 = pow(tmp4002, tmp4006);
    else
    {
      tmp4008 = modf(1.0/tmp4003, &tmp4007);
      if(tmp4008 > 0.5)
      {
        tmp4008 -= 1.0;
        tmp4007 += 1.0;
      }
      else if(tmp4008 < -0.5)
      {
        tmp4008 += 1.0;
        tmp4007 -= 1.0;
      }
      if(fabs(tmp4008) < 1e-10 && ((unsigned long)tmp4007 & 1))
      {
        tmp4004 = -pow(-tmp4002, tmp4005)*pow(tmp4002, tmp4006);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4002, tmp4003);
      }
    }
  }
  else
  {
    tmp4004 = pow(tmp4002, tmp4003);
  }
  if(isnan(tmp4004) || isinf(tmp4004))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4002, tmp4003);
  }tmp4009 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4004,"(r_init[100] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4009 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[100] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4009);
    }
  }
  (data->simulationInfo->realParameter[604] /* omega_c[100] PARAM */) = sqrt(tmp4009);
  TRACE_POP
}

/*
equation index: 13803
type: SIMPLE_ASSIGN
r_init[99] = r_min + 99.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13803};
  (data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (99.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13804
type: SIMPLE_ASSIGN
omega_c[99] = sqrt(G * Md / (r_init[99] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13804};
  modelica_real tmp4010;
  modelica_real tmp4011;
  modelica_real tmp4012;
  modelica_real tmp4013;
  modelica_real tmp4014;
  modelica_real tmp4015;
  modelica_real tmp4016;
  modelica_real tmp4017;
  modelica_real tmp4018;
  modelica_real tmp4019;
  tmp4010 = (data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */);
  tmp4011 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4012 = (tmp4010 * tmp4010) + (tmp4011 * tmp4011);
  tmp4013 = 1.5;
  if(tmp4012 < 0.0 && tmp4013 != 0.0)
  {
    tmp4015 = modf(tmp4013, &tmp4016);
    
    if(tmp4015 > 0.5)
    {
      tmp4015 -= 1.0;
      tmp4016 += 1.0;
    }
    else if(tmp4015 < -0.5)
    {
      tmp4015 += 1.0;
      tmp4016 -= 1.0;
    }
    
    if(fabs(tmp4015) < 1e-10)
      tmp4014 = pow(tmp4012, tmp4016);
    else
    {
      tmp4018 = modf(1.0/tmp4013, &tmp4017);
      if(tmp4018 > 0.5)
      {
        tmp4018 -= 1.0;
        tmp4017 += 1.0;
      }
      else if(tmp4018 < -0.5)
      {
        tmp4018 += 1.0;
        tmp4017 -= 1.0;
      }
      if(fabs(tmp4018) < 1e-10 && ((unsigned long)tmp4017 & 1))
      {
        tmp4014 = -pow(-tmp4012, tmp4015)*pow(tmp4012, tmp4016);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4012, tmp4013);
      }
    }
  }
  else
  {
    tmp4014 = pow(tmp4012, tmp4013);
  }
  if(isnan(tmp4014) || isinf(tmp4014))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4012, tmp4013);
  }tmp4019 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4014,"(r_init[99] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4019 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[99] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4019);
    }
  }
  (data->simulationInfo->realParameter[603] /* omega_c[99] PARAM */) = sqrt(tmp4019);
  TRACE_POP
}

/*
equation index: 13805
type: SIMPLE_ASSIGN
r_init[98] = r_min + 98.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13805};
  (data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (98.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13806
type: SIMPLE_ASSIGN
omega_c[98] = sqrt(G * Md / (r_init[98] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13806(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13806};
  modelica_real tmp4020;
  modelica_real tmp4021;
  modelica_real tmp4022;
  modelica_real tmp4023;
  modelica_real tmp4024;
  modelica_real tmp4025;
  modelica_real tmp4026;
  modelica_real tmp4027;
  modelica_real tmp4028;
  modelica_real tmp4029;
  tmp4020 = (data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */);
  tmp4021 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4022 = (tmp4020 * tmp4020) + (tmp4021 * tmp4021);
  tmp4023 = 1.5;
  if(tmp4022 < 0.0 && tmp4023 != 0.0)
  {
    tmp4025 = modf(tmp4023, &tmp4026);
    
    if(tmp4025 > 0.5)
    {
      tmp4025 -= 1.0;
      tmp4026 += 1.0;
    }
    else if(tmp4025 < -0.5)
    {
      tmp4025 += 1.0;
      tmp4026 -= 1.0;
    }
    
    if(fabs(tmp4025) < 1e-10)
      tmp4024 = pow(tmp4022, tmp4026);
    else
    {
      tmp4028 = modf(1.0/tmp4023, &tmp4027);
      if(tmp4028 > 0.5)
      {
        tmp4028 -= 1.0;
        tmp4027 += 1.0;
      }
      else if(tmp4028 < -0.5)
      {
        tmp4028 += 1.0;
        tmp4027 -= 1.0;
      }
      if(fabs(tmp4028) < 1e-10 && ((unsigned long)tmp4027 & 1))
      {
        tmp4024 = -pow(-tmp4022, tmp4025)*pow(tmp4022, tmp4026);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4022, tmp4023);
      }
    }
  }
  else
  {
    tmp4024 = pow(tmp4022, tmp4023);
  }
  if(isnan(tmp4024) || isinf(tmp4024))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4022, tmp4023);
  }tmp4029 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4024,"(r_init[98] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4029 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[98] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4029);
    }
  }
  (data->simulationInfo->realParameter[602] /* omega_c[98] PARAM */) = sqrt(tmp4029);
  TRACE_POP
}

/*
equation index: 13807
type: SIMPLE_ASSIGN
r_init[97] = r_min + 97.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13807};
  (data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (97.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13808
type: SIMPLE_ASSIGN
omega_c[97] = sqrt(G * Md / (r_init[97] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13808(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13808};
  modelica_real tmp4030;
  modelica_real tmp4031;
  modelica_real tmp4032;
  modelica_real tmp4033;
  modelica_real tmp4034;
  modelica_real tmp4035;
  modelica_real tmp4036;
  modelica_real tmp4037;
  modelica_real tmp4038;
  modelica_real tmp4039;
  tmp4030 = (data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */);
  tmp4031 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4032 = (tmp4030 * tmp4030) + (tmp4031 * tmp4031);
  tmp4033 = 1.5;
  if(tmp4032 < 0.0 && tmp4033 != 0.0)
  {
    tmp4035 = modf(tmp4033, &tmp4036);
    
    if(tmp4035 > 0.5)
    {
      tmp4035 -= 1.0;
      tmp4036 += 1.0;
    }
    else if(tmp4035 < -0.5)
    {
      tmp4035 += 1.0;
      tmp4036 -= 1.0;
    }
    
    if(fabs(tmp4035) < 1e-10)
      tmp4034 = pow(tmp4032, tmp4036);
    else
    {
      tmp4038 = modf(1.0/tmp4033, &tmp4037);
      if(tmp4038 > 0.5)
      {
        tmp4038 -= 1.0;
        tmp4037 += 1.0;
      }
      else if(tmp4038 < -0.5)
      {
        tmp4038 += 1.0;
        tmp4037 -= 1.0;
      }
      if(fabs(tmp4038) < 1e-10 && ((unsigned long)tmp4037 & 1))
      {
        tmp4034 = -pow(-tmp4032, tmp4035)*pow(tmp4032, tmp4036);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4032, tmp4033);
      }
    }
  }
  else
  {
    tmp4034 = pow(tmp4032, tmp4033);
  }
  if(isnan(tmp4034) || isinf(tmp4034))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4032, tmp4033);
  }tmp4039 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4034,"(r_init[97] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4039 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[97] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4039);
    }
  }
  (data->simulationInfo->realParameter[601] /* omega_c[97] PARAM */) = sqrt(tmp4039);
  TRACE_POP
}

/*
equation index: 13809
type: SIMPLE_ASSIGN
r_init[96] = r_min + 96.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13809};
  (data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (96.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13810
type: SIMPLE_ASSIGN
omega_c[96] = sqrt(G * Md / (r_init[96] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13810(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13810};
  modelica_real tmp4040;
  modelica_real tmp4041;
  modelica_real tmp4042;
  modelica_real tmp4043;
  modelica_real tmp4044;
  modelica_real tmp4045;
  modelica_real tmp4046;
  modelica_real tmp4047;
  modelica_real tmp4048;
  modelica_real tmp4049;
  tmp4040 = (data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */);
  tmp4041 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4042 = (tmp4040 * tmp4040) + (tmp4041 * tmp4041);
  tmp4043 = 1.5;
  if(tmp4042 < 0.0 && tmp4043 != 0.0)
  {
    tmp4045 = modf(tmp4043, &tmp4046);
    
    if(tmp4045 > 0.5)
    {
      tmp4045 -= 1.0;
      tmp4046 += 1.0;
    }
    else if(tmp4045 < -0.5)
    {
      tmp4045 += 1.0;
      tmp4046 -= 1.0;
    }
    
    if(fabs(tmp4045) < 1e-10)
      tmp4044 = pow(tmp4042, tmp4046);
    else
    {
      tmp4048 = modf(1.0/tmp4043, &tmp4047);
      if(tmp4048 > 0.5)
      {
        tmp4048 -= 1.0;
        tmp4047 += 1.0;
      }
      else if(tmp4048 < -0.5)
      {
        tmp4048 += 1.0;
        tmp4047 -= 1.0;
      }
      if(fabs(tmp4048) < 1e-10 && ((unsigned long)tmp4047 & 1))
      {
        tmp4044 = -pow(-tmp4042, tmp4045)*pow(tmp4042, tmp4046);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4042, tmp4043);
      }
    }
  }
  else
  {
    tmp4044 = pow(tmp4042, tmp4043);
  }
  if(isnan(tmp4044) || isinf(tmp4044))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4042, tmp4043);
  }tmp4049 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4044,"(r_init[96] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4049 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[96] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4049);
    }
  }
  (data->simulationInfo->realParameter[600] /* omega_c[96] PARAM */) = sqrt(tmp4049);
  TRACE_POP
}

/*
equation index: 13811
type: SIMPLE_ASSIGN
r_init[95] = r_min + 95.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13811};
  (data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (95.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13812
type: SIMPLE_ASSIGN
omega_c[95] = sqrt(G * Md / (r_init[95] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13812};
  modelica_real tmp4050;
  modelica_real tmp4051;
  modelica_real tmp4052;
  modelica_real tmp4053;
  modelica_real tmp4054;
  modelica_real tmp4055;
  modelica_real tmp4056;
  modelica_real tmp4057;
  modelica_real tmp4058;
  modelica_real tmp4059;
  tmp4050 = (data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */);
  tmp4051 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4052 = (tmp4050 * tmp4050) + (tmp4051 * tmp4051);
  tmp4053 = 1.5;
  if(tmp4052 < 0.0 && tmp4053 != 0.0)
  {
    tmp4055 = modf(tmp4053, &tmp4056);
    
    if(tmp4055 > 0.5)
    {
      tmp4055 -= 1.0;
      tmp4056 += 1.0;
    }
    else if(tmp4055 < -0.5)
    {
      tmp4055 += 1.0;
      tmp4056 -= 1.0;
    }
    
    if(fabs(tmp4055) < 1e-10)
      tmp4054 = pow(tmp4052, tmp4056);
    else
    {
      tmp4058 = modf(1.0/tmp4053, &tmp4057);
      if(tmp4058 > 0.5)
      {
        tmp4058 -= 1.0;
        tmp4057 += 1.0;
      }
      else if(tmp4058 < -0.5)
      {
        tmp4058 += 1.0;
        tmp4057 -= 1.0;
      }
      if(fabs(tmp4058) < 1e-10 && ((unsigned long)tmp4057 & 1))
      {
        tmp4054 = -pow(-tmp4052, tmp4055)*pow(tmp4052, tmp4056);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4052, tmp4053);
      }
    }
  }
  else
  {
    tmp4054 = pow(tmp4052, tmp4053);
  }
  if(isnan(tmp4054) || isinf(tmp4054))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4052, tmp4053);
  }tmp4059 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4054,"(r_init[95] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4059 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[95] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4059);
    }
  }
  (data->simulationInfo->realParameter[599] /* omega_c[95] PARAM */) = sqrt(tmp4059);
  TRACE_POP
}

/*
equation index: 13813
type: SIMPLE_ASSIGN
r_init[94] = r_min + 94.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13813};
  (data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (94.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13814
type: SIMPLE_ASSIGN
omega_c[94] = sqrt(G * Md / (r_init[94] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13814(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13814};
  modelica_real tmp4060;
  modelica_real tmp4061;
  modelica_real tmp4062;
  modelica_real tmp4063;
  modelica_real tmp4064;
  modelica_real tmp4065;
  modelica_real tmp4066;
  modelica_real tmp4067;
  modelica_real tmp4068;
  modelica_real tmp4069;
  tmp4060 = (data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */);
  tmp4061 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4062 = (tmp4060 * tmp4060) + (tmp4061 * tmp4061);
  tmp4063 = 1.5;
  if(tmp4062 < 0.0 && tmp4063 != 0.0)
  {
    tmp4065 = modf(tmp4063, &tmp4066);
    
    if(tmp4065 > 0.5)
    {
      tmp4065 -= 1.0;
      tmp4066 += 1.0;
    }
    else if(tmp4065 < -0.5)
    {
      tmp4065 += 1.0;
      tmp4066 -= 1.0;
    }
    
    if(fabs(tmp4065) < 1e-10)
      tmp4064 = pow(tmp4062, tmp4066);
    else
    {
      tmp4068 = modf(1.0/tmp4063, &tmp4067);
      if(tmp4068 > 0.5)
      {
        tmp4068 -= 1.0;
        tmp4067 += 1.0;
      }
      else if(tmp4068 < -0.5)
      {
        tmp4068 += 1.0;
        tmp4067 -= 1.0;
      }
      if(fabs(tmp4068) < 1e-10 && ((unsigned long)tmp4067 & 1))
      {
        tmp4064 = -pow(-tmp4062, tmp4065)*pow(tmp4062, tmp4066);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4062, tmp4063);
      }
    }
  }
  else
  {
    tmp4064 = pow(tmp4062, tmp4063);
  }
  if(isnan(tmp4064) || isinf(tmp4064))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4062, tmp4063);
  }tmp4069 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4064,"(r_init[94] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4069 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[94] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4069);
    }
  }
  (data->simulationInfo->realParameter[598] /* omega_c[94] PARAM */) = sqrt(tmp4069);
  TRACE_POP
}

/*
equation index: 13815
type: SIMPLE_ASSIGN
r_init[93] = r_min + 93.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13815};
  (data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (93.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13816
type: SIMPLE_ASSIGN
omega_c[93] = sqrt(G * Md / (r_init[93] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13816(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13816};
  modelica_real tmp4070;
  modelica_real tmp4071;
  modelica_real tmp4072;
  modelica_real tmp4073;
  modelica_real tmp4074;
  modelica_real tmp4075;
  modelica_real tmp4076;
  modelica_real tmp4077;
  modelica_real tmp4078;
  modelica_real tmp4079;
  tmp4070 = (data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */);
  tmp4071 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4072 = (tmp4070 * tmp4070) + (tmp4071 * tmp4071);
  tmp4073 = 1.5;
  if(tmp4072 < 0.0 && tmp4073 != 0.0)
  {
    tmp4075 = modf(tmp4073, &tmp4076);
    
    if(tmp4075 > 0.5)
    {
      tmp4075 -= 1.0;
      tmp4076 += 1.0;
    }
    else if(tmp4075 < -0.5)
    {
      tmp4075 += 1.0;
      tmp4076 -= 1.0;
    }
    
    if(fabs(tmp4075) < 1e-10)
      tmp4074 = pow(tmp4072, tmp4076);
    else
    {
      tmp4078 = modf(1.0/tmp4073, &tmp4077);
      if(tmp4078 > 0.5)
      {
        tmp4078 -= 1.0;
        tmp4077 += 1.0;
      }
      else if(tmp4078 < -0.5)
      {
        tmp4078 += 1.0;
        tmp4077 -= 1.0;
      }
      if(fabs(tmp4078) < 1e-10 && ((unsigned long)tmp4077 & 1))
      {
        tmp4074 = -pow(-tmp4072, tmp4075)*pow(tmp4072, tmp4076);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4072, tmp4073);
      }
    }
  }
  else
  {
    tmp4074 = pow(tmp4072, tmp4073);
  }
  if(isnan(tmp4074) || isinf(tmp4074))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4072, tmp4073);
  }tmp4079 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4074,"(r_init[93] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4079 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[93] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4079);
    }
  }
  (data->simulationInfo->realParameter[597] /* omega_c[93] PARAM */) = sqrt(tmp4079);
  TRACE_POP
}

/*
equation index: 13817
type: SIMPLE_ASSIGN
r_init[92] = r_min + 92.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13817};
  (data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (92.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13818
type: SIMPLE_ASSIGN
omega_c[92] = sqrt(G * Md / (r_init[92] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13818(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13818};
  modelica_real tmp4080;
  modelica_real tmp4081;
  modelica_real tmp4082;
  modelica_real tmp4083;
  modelica_real tmp4084;
  modelica_real tmp4085;
  modelica_real tmp4086;
  modelica_real tmp4087;
  modelica_real tmp4088;
  modelica_real tmp4089;
  tmp4080 = (data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */);
  tmp4081 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4082 = (tmp4080 * tmp4080) + (tmp4081 * tmp4081);
  tmp4083 = 1.5;
  if(tmp4082 < 0.0 && tmp4083 != 0.0)
  {
    tmp4085 = modf(tmp4083, &tmp4086);
    
    if(tmp4085 > 0.5)
    {
      tmp4085 -= 1.0;
      tmp4086 += 1.0;
    }
    else if(tmp4085 < -0.5)
    {
      tmp4085 += 1.0;
      tmp4086 -= 1.0;
    }
    
    if(fabs(tmp4085) < 1e-10)
      tmp4084 = pow(tmp4082, tmp4086);
    else
    {
      tmp4088 = modf(1.0/tmp4083, &tmp4087);
      if(tmp4088 > 0.5)
      {
        tmp4088 -= 1.0;
        tmp4087 += 1.0;
      }
      else if(tmp4088 < -0.5)
      {
        tmp4088 += 1.0;
        tmp4087 -= 1.0;
      }
      if(fabs(tmp4088) < 1e-10 && ((unsigned long)tmp4087 & 1))
      {
        tmp4084 = -pow(-tmp4082, tmp4085)*pow(tmp4082, tmp4086);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4082, tmp4083);
      }
    }
  }
  else
  {
    tmp4084 = pow(tmp4082, tmp4083);
  }
  if(isnan(tmp4084) || isinf(tmp4084))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4082, tmp4083);
  }tmp4089 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4084,"(r_init[92] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4089 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[92] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4089);
    }
  }
  (data->simulationInfo->realParameter[596] /* omega_c[92] PARAM */) = sqrt(tmp4089);
  TRACE_POP
}

/*
equation index: 13819
type: SIMPLE_ASSIGN
r_init[91] = r_min + 91.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13819};
  (data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (91.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13820
type: SIMPLE_ASSIGN
omega_c[91] = sqrt(G * Md / (r_init[91] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13820};
  modelica_real tmp4090;
  modelica_real tmp4091;
  modelica_real tmp4092;
  modelica_real tmp4093;
  modelica_real tmp4094;
  modelica_real tmp4095;
  modelica_real tmp4096;
  modelica_real tmp4097;
  modelica_real tmp4098;
  modelica_real tmp4099;
  tmp4090 = (data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */);
  tmp4091 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4092 = (tmp4090 * tmp4090) + (tmp4091 * tmp4091);
  tmp4093 = 1.5;
  if(tmp4092 < 0.0 && tmp4093 != 0.0)
  {
    tmp4095 = modf(tmp4093, &tmp4096);
    
    if(tmp4095 > 0.5)
    {
      tmp4095 -= 1.0;
      tmp4096 += 1.0;
    }
    else if(tmp4095 < -0.5)
    {
      tmp4095 += 1.0;
      tmp4096 -= 1.0;
    }
    
    if(fabs(tmp4095) < 1e-10)
      tmp4094 = pow(tmp4092, tmp4096);
    else
    {
      tmp4098 = modf(1.0/tmp4093, &tmp4097);
      if(tmp4098 > 0.5)
      {
        tmp4098 -= 1.0;
        tmp4097 += 1.0;
      }
      else if(tmp4098 < -0.5)
      {
        tmp4098 += 1.0;
        tmp4097 -= 1.0;
      }
      if(fabs(tmp4098) < 1e-10 && ((unsigned long)tmp4097 & 1))
      {
        tmp4094 = -pow(-tmp4092, tmp4095)*pow(tmp4092, tmp4096);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4092, tmp4093);
      }
    }
  }
  else
  {
    tmp4094 = pow(tmp4092, tmp4093);
  }
  if(isnan(tmp4094) || isinf(tmp4094))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4092, tmp4093);
  }tmp4099 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4094,"(r_init[91] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4099 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[91] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4099);
    }
  }
  (data->simulationInfo->realParameter[595] /* omega_c[91] PARAM */) = sqrt(tmp4099);
  TRACE_POP
}

/*
equation index: 13821
type: SIMPLE_ASSIGN
r_init[90] = r_min + 90.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13821};
  (data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (90.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13822
type: SIMPLE_ASSIGN
omega_c[90] = sqrt(G * Md / (r_init[90] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13822(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13822};
  modelica_real tmp4100;
  modelica_real tmp4101;
  modelica_real tmp4102;
  modelica_real tmp4103;
  modelica_real tmp4104;
  modelica_real tmp4105;
  modelica_real tmp4106;
  modelica_real tmp4107;
  modelica_real tmp4108;
  modelica_real tmp4109;
  tmp4100 = (data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */);
  tmp4101 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4102 = (tmp4100 * tmp4100) + (tmp4101 * tmp4101);
  tmp4103 = 1.5;
  if(tmp4102 < 0.0 && tmp4103 != 0.0)
  {
    tmp4105 = modf(tmp4103, &tmp4106);
    
    if(tmp4105 > 0.5)
    {
      tmp4105 -= 1.0;
      tmp4106 += 1.0;
    }
    else if(tmp4105 < -0.5)
    {
      tmp4105 += 1.0;
      tmp4106 -= 1.0;
    }
    
    if(fabs(tmp4105) < 1e-10)
      tmp4104 = pow(tmp4102, tmp4106);
    else
    {
      tmp4108 = modf(1.0/tmp4103, &tmp4107);
      if(tmp4108 > 0.5)
      {
        tmp4108 -= 1.0;
        tmp4107 += 1.0;
      }
      else if(tmp4108 < -0.5)
      {
        tmp4108 += 1.0;
        tmp4107 -= 1.0;
      }
      if(fabs(tmp4108) < 1e-10 && ((unsigned long)tmp4107 & 1))
      {
        tmp4104 = -pow(-tmp4102, tmp4105)*pow(tmp4102, tmp4106);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4102, tmp4103);
      }
    }
  }
  else
  {
    tmp4104 = pow(tmp4102, tmp4103);
  }
  if(isnan(tmp4104) || isinf(tmp4104))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4102, tmp4103);
  }tmp4109 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4104,"(r_init[90] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4109 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[90] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4109);
    }
  }
  (data->simulationInfo->realParameter[594] /* omega_c[90] PARAM */) = sqrt(tmp4109);
  TRACE_POP
}

/*
equation index: 13823
type: SIMPLE_ASSIGN
r_init[89] = r_min + 89.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13823};
  (data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (89.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13824
type: SIMPLE_ASSIGN
omega_c[89] = sqrt(G * Md / (r_init[89] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13824(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13824};
  modelica_real tmp4110;
  modelica_real tmp4111;
  modelica_real tmp4112;
  modelica_real tmp4113;
  modelica_real tmp4114;
  modelica_real tmp4115;
  modelica_real tmp4116;
  modelica_real tmp4117;
  modelica_real tmp4118;
  modelica_real tmp4119;
  tmp4110 = (data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */);
  tmp4111 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4112 = (tmp4110 * tmp4110) + (tmp4111 * tmp4111);
  tmp4113 = 1.5;
  if(tmp4112 < 0.0 && tmp4113 != 0.0)
  {
    tmp4115 = modf(tmp4113, &tmp4116);
    
    if(tmp4115 > 0.5)
    {
      tmp4115 -= 1.0;
      tmp4116 += 1.0;
    }
    else if(tmp4115 < -0.5)
    {
      tmp4115 += 1.0;
      tmp4116 -= 1.0;
    }
    
    if(fabs(tmp4115) < 1e-10)
      tmp4114 = pow(tmp4112, tmp4116);
    else
    {
      tmp4118 = modf(1.0/tmp4113, &tmp4117);
      if(tmp4118 > 0.5)
      {
        tmp4118 -= 1.0;
        tmp4117 += 1.0;
      }
      else if(tmp4118 < -0.5)
      {
        tmp4118 += 1.0;
        tmp4117 -= 1.0;
      }
      if(fabs(tmp4118) < 1e-10 && ((unsigned long)tmp4117 & 1))
      {
        tmp4114 = -pow(-tmp4112, tmp4115)*pow(tmp4112, tmp4116);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4112, tmp4113);
      }
    }
  }
  else
  {
    tmp4114 = pow(tmp4112, tmp4113);
  }
  if(isnan(tmp4114) || isinf(tmp4114))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4112, tmp4113);
  }tmp4119 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4114,"(r_init[89] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4119 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[89] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4119);
    }
  }
  (data->simulationInfo->realParameter[593] /* omega_c[89] PARAM */) = sqrt(tmp4119);
  TRACE_POP
}

/*
equation index: 13825
type: SIMPLE_ASSIGN
r_init[88] = r_min + 88.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13825};
  (data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (88.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13826
type: SIMPLE_ASSIGN
omega_c[88] = sqrt(G * Md / (r_init[88] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13826(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13826};
  modelica_real tmp4120;
  modelica_real tmp4121;
  modelica_real tmp4122;
  modelica_real tmp4123;
  modelica_real tmp4124;
  modelica_real tmp4125;
  modelica_real tmp4126;
  modelica_real tmp4127;
  modelica_real tmp4128;
  modelica_real tmp4129;
  tmp4120 = (data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */);
  tmp4121 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4122 = (tmp4120 * tmp4120) + (tmp4121 * tmp4121);
  tmp4123 = 1.5;
  if(tmp4122 < 0.0 && tmp4123 != 0.0)
  {
    tmp4125 = modf(tmp4123, &tmp4126);
    
    if(tmp4125 > 0.5)
    {
      tmp4125 -= 1.0;
      tmp4126 += 1.0;
    }
    else if(tmp4125 < -0.5)
    {
      tmp4125 += 1.0;
      tmp4126 -= 1.0;
    }
    
    if(fabs(tmp4125) < 1e-10)
      tmp4124 = pow(tmp4122, tmp4126);
    else
    {
      tmp4128 = modf(1.0/tmp4123, &tmp4127);
      if(tmp4128 > 0.5)
      {
        tmp4128 -= 1.0;
        tmp4127 += 1.0;
      }
      else if(tmp4128 < -0.5)
      {
        tmp4128 += 1.0;
        tmp4127 -= 1.0;
      }
      if(fabs(tmp4128) < 1e-10 && ((unsigned long)tmp4127 & 1))
      {
        tmp4124 = -pow(-tmp4122, tmp4125)*pow(tmp4122, tmp4126);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4122, tmp4123);
      }
    }
  }
  else
  {
    tmp4124 = pow(tmp4122, tmp4123);
  }
  if(isnan(tmp4124) || isinf(tmp4124))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4122, tmp4123);
  }tmp4129 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4124,"(r_init[88] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4129 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[88] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4129);
    }
  }
  (data->simulationInfo->realParameter[592] /* omega_c[88] PARAM */) = sqrt(tmp4129);
  TRACE_POP
}

/*
equation index: 13827
type: SIMPLE_ASSIGN
r_init[87] = r_min + 87.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13827};
  (data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (87.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13828
type: SIMPLE_ASSIGN
omega_c[87] = sqrt(G * Md / (r_init[87] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13828};
  modelica_real tmp4130;
  modelica_real tmp4131;
  modelica_real tmp4132;
  modelica_real tmp4133;
  modelica_real tmp4134;
  modelica_real tmp4135;
  modelica_real tmp4136;
  modelica_real tmp4137;
  modelica_real tmp4138;
  modelica_real tmp4139;
  tmp4130 = (data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */);
  tmp4131 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4132 = (tmp4130 * tmp4130) + (tmp4131 * tmp4131);
  tmp4133 = 1.5;
  if(tmp4132 < 0.0 && tmp4133 != 0.0)
  {
    tmp4135 = modf(tmp4133, &tmp4136);
    
    if(tmp4135 > 0.5)
    {
      tmp4135 -= 1.0;
      tmp4136 += 1.0;
    }
    else if(tmp4135 < -0.5)
    {
      tmp4135 += 1.0;
      tmp4136 -= 1.0;
    }
    
    if(fabs(tmp4135) < 1e-10)
      tmp4134 = pow(tmp4132, tmp4136);
    else
    {
      tmp4138 = modf(1.0/tmp4133, &tmp4137);
      if(tmp4138 > 0.5)
      {
        tmp4138 -= 1.0;
        tmp4137 += 1.0;
      }
      else if(tmp4138 < -0.5)
      {
        tmp4138 += 1.0;
        tmp4137 -= 1.0;
      }
      if(fabs(tmp4138) < 1e-10 && ((unsigned long)tmp4137 & 1))
      {
        tmp4134 = -pow(-tmp4132, tmp4135)*pow(tmp4132, tmp4136);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4132, tmp4133);
      }
    }
  }
  else
  {
    tmp4134 = pow(tmp4132, tmp4133);
  }
  if(isnan(tmp4134) || isinf(tmp4134))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4132, tmp4133);
  }tmp4139 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4134,"(r_init[87] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4139 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[87] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4139);
    }
  }
  (data->simulationInfo->realParameter[591] /* omega_c[87] PARAM */) = sqrt(tmp4139);
  TRACE_POP
}

/*
equation index: 13829
type: SIMPLE_ASSIGN
r_init[86] = r_min + 86.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13829};
  (data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (86.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13830
type: SIMPLE_ASSIGN
omega_c[86] = sqrt(G * Md / (r_init[86] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13830(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13830};
  modelica_real tmp4140;
  modelica_real tmp4141;
  modelica_real tmp4142;
  modelica_real tmp4143;
  modelica_real tmp4144;
  modelica_real tmp4145;
  modelica_real tmp4146;
  modelica_real tmp4147;
  modelica_real tmp4148;
  modelica_real tmp4149;
  tmp4140 = (data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */);
  tmp4141 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4142 = (tmp4140 * tmp4140) + (tmp4141 * tmp4141);
  tmp4143 = 1.5;
  if(tmp4142 < 0.0 && tmp4143 != 0.0)
  {
    tmp4145 = modf(tmp4143, &tmp4146);
    
    if(tmp4145 > 0.5)
    {
      tmp4145 -= 1.0;
      tmp4146 += 1.0;
    }
    else if(tmp4145 < -0.5)
    {
      tmp4145 += 1.0;
      tmp4146 -= 1.0;
    }
    
    if(fabs(tmp4145) < 1e-10)
      tmp4144 = pow(tmp4142, tmp4146);
    else
    {
      tmp4148 = modf(1.0/tmp4143, &tmp4147);
      if(tmp4148 > 0.5)
      {
        tmp4148 -= 1.0;
        tmp4147 += 1.0;
      }
      else if(tmp4148 < -0.5)
      {
        tmp4148 += 1.0;
        tmp4147 -= 1.0;
      }
      if(fabs(tmp4148) < 1e-10 && ((unsigned long)tmp4147 & 1))
      {
        tmp4144 = -pow(-tmp4142, tmp4145)*pow(tmp4142, tmp4146);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4142, tmp4143);
      }
    }
  }
  else
  {
    tmp4144 = pow(tmp4142, tmp4143);
  }
  if(isnan(tmp4144) || isinf(tmp4144))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4142, tmp4143);
  }tmp4149 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4144,"(r_init[86] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4149 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[86] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4149);
    }
  }
  (data->simulationInfo->realParameter[590] /* omega_c[86] PARAM */) = sqrt(tmp4149);
  TRACE_POP
}

/*
equation index: 13831
type: SIMPLE_ASSIGN
r_init[85] = r_min + 85.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13831};
  (data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (85.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13832
type: SIMPLE_ASSIGN
omega_c[85] = sqrt(G * Md / (r_init[85] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13832(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13832};
  modelica_real tmp4150;
  modelica_real tmp4151;
  modelica_real tmp4152;
  modelica_real tmp4153;
  modelica_real tmp4154;
  modelica_real tmp4155;
  modelica_real tmp4156;
  modelica_real tmp4157;
  modelica_real tmp4158;
  modelica_real tmp4159;
  tmp4150 = (data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */);
  tmp4151 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4152 = (tmp4150 * tmp4150) + (tmp4151 * tmp4151);
  tmp4153 = 1.5;
  if(tmp4152 < 0.0 && tmp4153 != 0.0)
  {
    tmp4155 = modf(tmp4153, &tmp4156);
    
    if(tmp4155 > 0.5)
    {
      tmp4155 -= 1.0;
      tmp4156 += 1.0;
    }
    else if(tmp4155 < -0.5)
    {
      tmp4155 += 1.0;
      tmp4156 -= 1.0;
    }
    
    if(fabs(tmp4155) < 1e-10)
      tmp4154 = pow(tmp4152, tmp4156);
    else
    {
      tmp4158 = modf(1.0/tmp4153, &tmp4157);
      if(tmp4158 > 0.5)
      {
        tmp4158 -= 1.0;
        tmp4157 += 1.0;
      }
      else if(tmp4158 < -0.5)
      {
        tmp4158 += 1.0;
        tmp4157 -= 1.0;
      }
      if(fabs(tmp4158) < 1e-10 && ((unsigned long)tmp4157 & 1))
      {
        tmp4154 = -pow(-tmp4152, tmp4155)*pow(tmp4152, tmp4156);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4152, tmp4153);
      }
    }
  }
  else
  {
    tmp4154 = pow(tmp4152, tmp4153);
  }
  if(isnan(tmp4154) || isinf(tmp4154))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4152, tmp4153);
  }tmp4159 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4154,"(r_init[85] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4159 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[85] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4159);
    }
  }
  (data->simulationInfo->realParameter[589] /* omega_c[85] PARAM */) = sqrt(tmp4159);
  TRACE_POP
}

/*
equation index: 13833
type: SIMPLE_ASSIGN
r_init[84] = r_min + 84.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13833};
  (data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (84.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13834
type: SIMPLE_ASSIGN
omega_c[84] = sqrt(G * Md / (r_init[84] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13834(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13834};
  modelica_real tmp4160;
  modelica_real tmp4161;
  modelica_real tmp4162;
  modelica_real tmp4163;
  modelica_real tmp4164;
  modelica_real tmp4165;
  modelica_real tmp4166;
  modelica_real tmp4167;
  modelica_real tmp4168;
  modelica_real tmp4169;
  tmp4160 = (data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */);
  tmp4161 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4162 = (tmp4160 * tmp4160) + (tmp4161 * tmp4161);
  tmp4163 = 1.5;
  if(tmp4162 < 0.0 && tmp4163 != 0.0)
  {
    tmp4165 = modf(tmp4163, &tmp4166);
    
    if(tmp4165 > 0.5)
    {
      tmp4165 -= 1.0;
      tmp4166 += 1.0;
    }
    else if(tmp4165 < -0.5)
    {
      tmp4165 += 1.0;
      tmp4166 -= 1.0;
    }
    
    if(fabs(tmp4165) < 1e-10)
      tmp4164 = pow(tmp4162, tmp4166);
    else
    {
      tmp4168 = modf(1.0/tmp4163, &tmp4167);
      if(tmp4168 > 0.5)
      {
        tmp4168 -= 1.0;
        tmp4167 += 1.0;
      }
      else if(tmp4168 < -0.5)
      {
        tmp4168 += 1.0;
        tmp4167 -= 1.0;
      }
      if(fabs(tmp4168) < 1e-10 && ((unsigned long)tmp4167 & 1))
      {
        tmp4164 = -pow(-tmp4162, tmp4165)*pow(tmp4162, tmp4166);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4162, tmp4163);
      }
    }
  }
  else
  {
    tmp4164 = pow(tmp4162, tmp4163);
  }
  if(isnan(tmp4164) || isinf(tmp4164))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4162, tmp4163);
  }tmp4169 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4164,"(r_init[84] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4169 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[84] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4169);
    }
  }
  (data->simulationInfo->realParameter[588] /* omega_c[84] PARAM */) = sqrt(tmp4169);
  TRACE_POP
}

/*
equation index: 13835
type: SIMPLE_ASSIGN
r_init[83] = r_min + 83.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13835};
  (data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (83.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13836
type: SIMPLE_ASSIGN
omega_c[83] = sqrt(G * Md / (r_init[83] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13836};
  modelica_real tmp4170;
  modelica_real tmp4171;
  modelica_real tmp4172;
  modelica_real tmp4173;
  modelica_real tmp4174;
  modelica_real tmp4175;
  modelica_real tmp4176;
  modelica_real tmp4177;
  modelica_real tmp4178;
  modelica_real tmp4179;
  tmp4170 = (data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */);
  tmp4171 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4172 = (tmp4170 * tmp4170) + (tmp4171 * tmp4171);
  tmp4173 = 1.5;
  if(tmp4172 < 0.0 && tmp4173 != 0.0)
  {
    tmp4175 = modf(tmp4173, &tmp4176);
    
    if(tmp4175 > 0.5)
    {
      tmp4175 -= 1.0;
      tmp4176 += 1.0;
    }
    else if(tmp4175 < -0.5)
    {
      tmp4175 += 1.0;
      tmp4176 -= 1.0;
    }
    
    if(fabs(tmp4175) < 1e-10)
      tmp4174 = pow(tmp4172, tmp4176);
    else
    {
      tmp4178 = modf(1.0/tmp4173, &tmp4177);
      if(tmp4178 > 0.5)
      {
        tmp4178 -= 1.0;
        tmp4177 += 1.0;
      }
      else if(tmp4178 < -0.5)
      {
        tmp4178 += 1.0;
        tmp4177 -= 1.0;
      }
      if(fabs(tmp4178) < 1e-10 && ((unsigned long)tmp4177 & 1))
      {
        tmp4174 = -pow(-tmp4172, tmp4175)*pow(tmp4172, tmp4176);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4172, tmp4173);
      }
    }
  }
  else
  {
    tmp4174 = pow(tmp4172, tmp4173);
  }
  if(isnan(tmp4174) || isinf(tmp4174))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4172, tmp4173);
  }tmp4179 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4174,"(r_init[83] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4179 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[83] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4179);
    }
  }
  (data->simulationInfo->realParameter[587] /* omega_c[83] PARAM */) = sqrt(tmp4179);
  TRACE_POP
}

/*
equation index: 13837
type: SIMPLE_ASSIGN
r_init[82] = r_min + 82.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13837};
  (data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (82.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13838
type: SIMPLE_ASSIGN
omega_c[82] = sqrt(G * Md / (r_init[82] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13838(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13838};
  modelica_real tmp4180;
  modelica_real tmp4181;
  modelica_real tmp4182;
  modelica_real tmp4183;
  modelica_real tmp4184;
  modelica_real tmp4185;
  modelica_real tmp4186;
  modelica_real tmp4187;
  modelica_real tmp4188;
  modelica_real tmp4189;
  tmp4180 = (data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */);
  tmp4181 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4182 = (tmp4180 * tmp4180) + (tmp4181 * tmp4181);
  tmp4183 = 1.5;
  if(tmp4182 < 0.0 && tmp4183 != 0.0)
  {
    tmp4185 = modf(tmp4183, &tmp4186);
    
    if(tmp4185 > 0.5)
    {
      tmp4185 -= 1.0;
      tmp4186 += 1.0;
    }
    else if(tmp4185 < -0.5)
    {
      tmp4185 += 1.0;
      tmp4186 -= 1.0;
    }
    
    if(fabs(tmp4185) < 1e-10)
      tmp4184 = pow(tmp4182, tmp4186);
    else
    {
      tmp4188 = modf(1.0/tmp4183, &tmp4187);
      if(tmp4188 > 0.5)
      {
        tmp4188 -= 1.0;
        tmp4187 += 1.0;
      }
      else if(tmp4188 < -0.5)
      {
        tmp4188 += 1.0;
        tmp4187 -= 1.0;
      }
      if(fabs(tmp4188) < 1e-10 && ((unsigned long)tmp4187 & 1))
      {
        tmp4184 = -pow(-tmp4182, tmp4185)*pow(tmp4182, tmp4186);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4182, tmp4183);
      }
    }
  }
  else
  {
    tmp4184 = pow(tmp4182, tmp4183);
  }
  if(isnan(tmp4184) || isinf(tmp4184))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4182, tmp4183);
  }tmp4189 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4184,"(r_init[82] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4189 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[82] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4189);
    }
  }
  (data->simulationInfo->realParameter[586] /* omega_c[82] PARAM */) = sqrt(tmp4189);
  TRACE_POP
}

/*
equation index: 13839
type: SIMPLE_ASSIGN
r_init[81] = r_min + 81.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13839};
  (data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (81.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13840
type: SIMPLE_ASSIGN
omega_c[81] = sqrt(G * Md / (r_init[81] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13840(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13840};
  modelica_real tmp4190;
  modelica_real tmp4191;
  modelica_real tmp4192;
  modelica_real tmp4193;
  modelica_real tmp4194;
  modelica_real tmp4195;
  modelica_real tmp4196;
  modelica_real tmp4197;
  modelica_real tmp4198;
  modelica_real tmp4199;
  tmp4190 = (data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */);
  tmp4191 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4192 = (tmp4190 * tmp4190) + (tmp4191 * tmp4191);
  tmp4193 = 1.5;
  if(tmp4192 < 0.0 && tmp4193 != 0.0)
  {
    tmp4195 = modf(tmp4193, &tmp4196);
    
    if(tmp4195 > 0.5)
    {
      tmp4195 -= 1.0;
      tmp4196 += 1.0;
    }
    else if(tmp4195 < -0.5)
    {
      tmp4195 += 1.0;
      tmp4196 -= 1.0;
    }
    
    if(fabs(tmp4195) < 1e-10)
      tmp4194 = pow(tmp4192, tmp4196);
    else
    {
      tmp4198 = modf(1.0/tmp4193, &tmp4197);
      if(tmp4198 > 0.5)
      {
        tmp4198 -= 1.0;
        tmp4197 += 1.0;
      }
      else if(tmp4198 < -0.5)
      {
        tmp4198 += 1.0;
        tmp4197 -= 1.0;
      }
      if(fabs(tmp4198) < 1e-10 && ((unsigned long)tmp4197 & 1))
      {
        tmp4194 = -pow(-tmp4192, tmp4195)*pow(tmp4192, tmp4196);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4192, tmp4193);
      }
    }
  }
  else
  {
    tmp4194 = pow(tmp4192, tmp4193);
  }
  if(isnan(tmp4194) || isinf(tmp4194))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4192, tmp4193);
  }tmp4199 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4194,"(r_init[81] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4199 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[81] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4199);
    }
  }
  (data->simulationInfo->realParameter[585] /* omega_c[81] PARAM */) = sqrt(tmp4199);
  TRACE_POP
}

/*
equation index: 13841
type: SIMPLE_ASSIGN
r_init[80] = r_min + 80.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13841};
  (data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (80.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13842
type: SIMPLE_ASSIGN
omega_c[80] = sqrt(G * Md / (r_init[80] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13842(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13842};
  modelica_real tmp4200;
  modelica_real tmp4201;
  modelica_real tmp4202;
  modelica_real tmp4203;
  modelica_real tmp4204;
  modelica_real tmp4205;
  modelica_real tmp4206;
  modelica_real tmp4207;
  modelica_real tmp4208;
  modelica_real tmp4209;
  tmp4200 = (data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */);
  tmp4201 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4202 = (tmp4200 * tmp4200) + (tmp4201 * tmp4201);
  tmp4203 = 1.5;
  if(tmp4202 < 0.0 && tmp4203 != 0.0)
  {
    tmp4205 = modf(tmp4203, &tmp4206);
    
    if(tmp4205 > 0.5)
    {
      tmp4205 -= 1.0;
      tmp4206 += 1.0;
    }
    else if(tmp4205 < -0.5)
    {
      tmp4205 += 1.0;
      tmp4206 -= 1.0;
    }
    
    if(fabs(tmp4205) < 1e-10)
      tmp4204 = pow(tmp4202, tmp4206);
    else
    {
      tmp4208 = modf(1.0/tmp4203, &tmp4207);
      if(tmp4208 > 0.5)
      {
        tmp4208 -= 1.0;
        tmp4207 += 1.0;
      }
      else if(tmp4208 < -0.5)
      {
        tmp4208 += 1.0;
        tmp4207 -= 1.0;
      }
      if(fabs(tmp4208) < 1e-10 && ((unsigned long)tmp4207 & 1))
      {
        tmp4204 = -pow(-tmp4202, tmp4205)*pow(tmp4202, tmp4206);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4202, tmp4203);
      }
    }
  }
  else
  {
    tmp4204 = pow(tmp4202, tmp4203);
  }
  if(isnan(tmp4204) || isinf(tmp4204))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4202, tmp4203);
  }tmp4209 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4204,"(r_init[80] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4209 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[80] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4209);
    }
  }
  (data->simulationInfo->realParameter[584] /* omega_c[80] PARAM */) = sqrt(tmp4209);
  TRACE_POP
}

/*
equation index: 13843
type: SIMPLE_ASSIGN
r_init[79] = r_min + 79.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13843};
  (data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (79.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13844
type: SIMPLE_ASSIGN
omega_c[79] = sqrt(G * Md / (r_init[79] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13844};
  modelica_real tmp4210;
  modelica_real tmp4211;
  modelica_real tmp4212;
  modelica_real tmp4213;
  modelica_real tmp4214;
  modelica_real tmp4215;
  modelica_real tmp4216;
  modelica_real tmp4217;
  modelica_real tmp4218;
  modelica_real tmp4219;
  tmp4210 = (data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */);
  tmp4211 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4212 = (tmp4210 * tmp4210) + (tmp4211 * tmp4211);
  tmp4213 = 1.5;
  if(tmp4212 < 0.0 && tmp4213 != 0.0)
  {
    tmp4215 = modf(tmp4213, &tmp4216);
    
    if(tmp4215 > 0.5)
    {
      tmp4215 -= 1.0;
      tmp4216 += 1.0;
    }
    else if(tmp4215 < -0.5)
    {
      tmp4215 += 1.0;
      tmp4216 -= 1.0;
    }
    
    if(fabs(tmp4215) < 1e-10)
      tmp4214 = pow(tmp4212, tmp4216);
    else
    {
      tmp4218 = modf(1.0/tmp4213, &tmp4217);
      if(tmp4218 > 0.5)
      {
        tmp4218 -= 1.0;
        tmp4217 += 1.0;
      }
      else if(tmp4218 < -0.5)
      {
        tmp4218 += 1.0;
        tmp4217 -= 1.0;
      }
      if(fabs(tmp4218) < 1e-10 && ((unsigned long)tmp4217 & 1))
      {
        tmp4214 = -pow(-tmp4212, tmp4215)*pow(tmp4212, tmp4216);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4212, tmp4213);
      }
    }
  }
  else
  {
    tmp4214 = pow(tmp4212, tmp4213);
  }
  if(isnan(tmp4214) || isinf(tmp4214))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4212, tmp4213);
  }tmp4219 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4214,"(r_init[79] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4219 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[79] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4219);
    }
  }
  (data->simulationInfo->realParameter[583] /* omega_c[79] PARAM */) = sqrt(tmp4219);
  TRACE_POP
}

/*
equation index: 13845
type: SIMPLE_ASSIGN
r_init[78] = r_min + 78.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13845};
  (data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (78.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13846
type: SIMPLE_ASSIGN
omega_c[78] = sqrt(G * Md / (r_init[78] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13846(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13846};
  modelica_real tmp4220;
  modelica_real tmp4221;
  modelica_real tmp4222;
  modelica_real tmp4223;
  modelica_real tmp4224;
  modelica_real tmp4225;
  modelica_real tmp4226;
  modelica_real tmp4227;
  modelica_real tmp4228;
  modelica_real tmp4229;
  tmp4220 = (data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */);
  tmp4221 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4222 = (tmp4220 * tmp4220) + (tmp4221 * tmp4221);
  tmp4223 = 1.5;
  if(tmp4222 < 0.0 && tmp4223 != 0.0)
  {
    tmp4225 = modf(tmp4223, &tmp4226);
    
    if(tmp4225 > 0.5)
    {
      tmp4225 -= 1.0;
      tmp4226 += 1.0;
    }
    else if(tmp4225 < -0.5)
    {
      tmp4225 += 1.0;
      tmp4226 -= 1.0;
    }
    
    if(fabs(tmp4225) < 1e-10)
      tmp4224 = pow(tmp4222, tmp4226);
    else
    {
      tmp4228 = modf(1.0/tmp4223, &tmp4227);
      if(tmp4228 > 0.5)
      {
        tmp4228 -= 1.0;
        tmp4227 += 1.0;
      }
      else if(tmp4228 < -0.5)
      {
        tmp4228 += 1.0;
        tmp4227 -= 1.0;
      }
      if(fabs(tmp4228) < 1e-10 && ((unsigned long)tmp4227 & 1))
      {
        tmp4224 = -pow(-tmp4222, tmp4225)*pow(tmp4222, tmp4226);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4222, tmp4223);
      }
    }
  }
  else
  {
    tmp4224 = pow(tmp4222, tmp4223);
  }
  if(isnan(tmp4224) || isinf(tmp4224))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4222, tmp4223);
  }tmp4229 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4224,"(r_init[78] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4229 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[78] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4229);
    }
  }
  (data->simulationInfo->realParameter[582] /* omega_c[78] PARAM */) = sqrt(tmp4229);
  TRACE_POP
}

/*
equation index: 13847
type: SIMPLE_ASSIGN
r_init[77] = r_min + 77.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13847};
  (data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (77.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13848
type: SIMPLE_ASSIGN
omega_c[77] = sqrt(G * Md / (r_init[77] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13848(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13848};
  modelica_real tmp4230;
  modelica_real tmp4231;
  modelica_real tmp4232;
  modelica_real tmp4233;
  modelica_real tmp4234;
  modelica_real tmp4235;
  modelica_real tmp4236;
  modelica_real tmp4237;
  modelica_real tmp4238;
  modelica_real tmp4239;
  tmp4230 = (data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */);
  tmp4231 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4232 = (tmp4230 * tmp4230) + (tmp4231 * tmp4231);
  tmp4233 = 1.5;
  if(tmp4232 < 0.0 && tmp4233 != 0.0)
  {
    tmp4235 = modf(tmp4233, &tmp4236);
    
    if(tmp4235 > 0.5)
    {
      tmp4235 -= 1.0;
      tmp4236 += 1.0;
    }
    else if(tmp4235 < -0.5)
    {
      tmp4235 += 1.0;
      tmp4236 -= 1.0;
    }
    
    if(fabs(tmp4235) < 1e-10)
      tmp4234 = pow(tmp4232, tmp4236);
    else
    {
      tmp4238 = modf(1.0/tmp4233, &tmp4237);
      if(tmp4238 > 0.5)
      {
        tmp4238 -= 1.0;
        tmp4237 += 1.0;
      }
      else if(tmp4238 < -0.5)
      {
        tmp4238 += 1.0;
        tmp4237 -= 1.0;
      }
      if(fabs(tmp4238) < 1e-10 && ((unsigned long)tmp4237 & 1))
      {
        tmp4234 = -pow(-tmp4232, tmp4235)*pow(tmp4232, tmp4236);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4232, tmp4233);
      }
    }
  }
  else
  {
    tmp4234 = pow(tmp4232, tmp4233);
  }
  if(isnan(tmp4234) || isinf(tmp4234))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4232, tmp4233);
  }tmp4239 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4234,"(r_init[77] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4239 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[77] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4239);
    }
  }
  (data->simulationInfo->realParameter[581] /* omega_c[77] PARAM */) = sqrt(tmp4239);
  TRACE_POP
}

/*
equation index: 13849
type: SIMPLE_ASSIGN
r_init[76] = r_min + 76.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13849};
  (data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (76.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13850
type: SIMPLE_ASSIGN
omega_c[76] = sqrt(G * Md / (r_init[76] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13850(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13850};
  modelica_real tmp4240;
  modelica_real tmp4241;
  modelica_real tmp4242;
  modelica_real tmp4243;
  modelica_real tmp4244;
  modelica_real tmp4245;
  modelica_real tmp4246;
  modelica_real tmp4247;
  modelica_real tmp4248;
  modelica_real tmp4249;
  tmp4240 = (data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */);
  tmp4241 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4242 = (tmp4240 * tmp4240) + (tmp4241 * tmp4241);
  tmp4243 = 1.5;
  if(tmp4242 < 0.0 && tmp4243 != 0.0)
  {
    tmp4245 = modf(tmp4243, &tmp4246);
    
    if(tmp4245 > 0.5)
    {
      tmp4245 -= 1.0;
      tmp4246 += 1.0;
    }
    else if(tmp4245 < -0.5)
    {
      tmp4245 += 1.0;
      tmp4246 -= 1.0;
    }
    
    if(fabs(tmp4245) < 1e-10)
      tmp4244 = pow(tmp4242, tmp4246);
    else
    {
      tmp4248 = modf(1.0/tmp4243, &tmp4247);
      if(tmp4248 > 0.5)
      {
        tmp4248 -= 1.0;
        tmp4247 += 1.0;
      }
      else if(tmp4248 < -0.5)
      {
        tmp4248 += 1.0;
        tmp4247 -= 1.0;
      }
      if(fabs(tmp4248) < 1e-10 && ((unsigned long)tmp4247 & 1))
      {
        tmp4244 = -pow(-tmp4242, tmp4245)*pow(tmp4242, tmp4246);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4242, tmp4243);
      }
    }
  }
  else
  {
    tmp4244 = pow(tmp4242, tmp4243);
  }
  if(isnan(tmp4244) || isinf(tmp4244))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4242, tmp4243);
  }tmp4249 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4244,"(r_init[76] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4249 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[76] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4249);
    }
  }
  (data->simulationInfo->realParameter[580] /* omega_c[76] PARAM */) = sqrt(tmp4249);
  TRACE_POP
}

/*
equation index: 13851
type: SIMPLE_ASSIGN
r_init[75] = r_min + 75.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13851};
  (data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (75.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13852
type: SIMPLE_ASSIGN
omega_c[75] = sqrt(G * Md / (r_init[75] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13852};
  modelica_real tmp4250;
  modelica_real tmp4251;
  modelica_real tmp4252;
  modelica_real tmp4253;
  modelica_real tmp4254;
  modelica_real tmp4255;
  modelica_real tmp4256;
  modelica_real tmp4257;
  modelica_real tmp4258;
  modelica_real tmp4259;
  tmp4250 = (data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */);
  tmp4251 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4252 = (tmp4250 * tmp4250) + (tmp4251 * tmp4251);
  tmp4253 = 1.5;
  if(tmp4252 < 0.0 && tmp4253 != 0.0)
  {
    tmp4255 = modf(tmp4253, &tmp4256);
    
    if(tmp4255 > 0.5)
    {
      tmp4255 -= 1.0;
      tmp4256 += 1.0;
    }
    else if(tmp4255 < -0.5)
    {
      tmp4255 += 1.0;
      tmp4256 -= 1.0;
    }
    
    if(fabs(tmp4255) < 1e-10)
      tmp4254 = pow(tmp4252, tmp4256);
    else
    {
      tmp4258 = modf(1.0/tmp4253, &tmp4257);
      if(tmp4258 > 0.5)
      {
        tmp4258 -= 1.0;
        tmp4257 += 1.0;
      }
      else if(tmp4258 < -0.5)
      {
        tmp4258 += 1.0;
        tmp4257 -= 1.0;
      }
      if(fabs(tmp4258) < 1e-10 && ((unsigned long)tmp4257 & 1))
      {
        tmp4254 = -pow(-tmp4252, tmp4255)*pow(tmp4252, tmp4256);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4252, tmp4253);
      }
    }
  }
  else
  {
    tmp4254 = pow(tmp4252, tmp4253);
  }
  if(isnan(tmp4254) || isinf(tmp4254))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4252, tmp4253);
  }tmp4259 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4254,"(r_init[75] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4259 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[75] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4259);
    }
  }
  (data->simulationInfo->realParameter[579] /* omega_c[75] PARAM */) = sqrt(tmp4259);
  TRACE_POP
}

/*
equation index: 13853
type: SIMPLE_ASSIGN
r_init[74] = r_min + 74.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13853};
  (data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (74.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13854
type: SIMPLE_ASSIGN
omega_c[74] = sqrt(G * Md / (r_init[74] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13854(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13854};
  modelica_real tmp4260;
  modelica_real tmp4261;
  modelica_real tmp4262;
  modelica_real tmp4263;
  modelica_real tmp4264;
  modelica_real tmp4265;
  modelica_real tmp4266;
  modelica_real tmp4267;
  modelica_real tmp4268;
  modelica_real tmp4269;
  tmp4260 = (data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */);
  tmp4261 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4262 = (tmp4260 * tmp4260) + (tmp4261 * tmp4261);
  tmp4263 = 1.5;
  if(tmp4262 < 0.0 && tmp4263 != 0.0)
  {
    tmp4265 = modf(tmp4263, &tmp4266);
    
    if(tmp4265 > 0.5)
    {
      tmp4265 -= 1.0;
      tmp4266 += 1.0;
    }
    else if(tmp4265 < -0.5)
    {
      tmp4265 += 1.0;
      tmp4266 -= 1.0;
    }
    
    if(fabs(tmp4265) < 1e-10)
      tmp4264 = pow(tmp4262, tmp4266);
    else
    {
      tmp4268 = modf(1.0/tmp4263, &tmp4267);
      if(tmp4268 > 0.5)
      {
        tmp4268 -= 1.0;
        tmp4267 += 1.0;
      }
      else if(tmp4268 < -0.5)
      {
        tmp4268 += 1.0;
        tmp4267 -= 1.0;
      }
      if(fabs(tmp4268) < 1e-10 && ((unsigned long)tmp4267 & 1))
      {
        tmp4264 = -pow(-tmp4262, tmp4265)*pow(tmp4262, tmp4266);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4262, tmp4263);
      }
    }
  }
  else
  {
    tmp4264 = pow(tmp4262, tmp4263);
  }
  if(isnan(tmp4264) || isinf(tmp4264))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4262, tmp4263);
  }tmp4269 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4264,"(r_init[74] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4269 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[74] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4269);
    }
  }
  (data->simulationInfo->realParameter[578] /* omega_c[74] PARAM */) = sqrt(tmp4269);
  TRACE_POP
}

/*
equation index: 13855
type: SIMPLE_ASSIGN
r_init[73] = r_min + 73.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13855};
  (data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (73.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13856
type: SIMPLE_ASSIGN
omega_c[73] = sqrt(G * Md / (r_init[73] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13856(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13856};
  modelica_real tmp4270;
  modelica_real tmp4271;
  modelica_real tmp4272;
  modelica_real tmp4273;
  modelica_real tmp4274;
  modelica_real tmp4275;
  modelica_real tmp4276;
  modelica_real tmp4277;
  modelica_real tmp4278;
  modelica_real tmp4279;
  tmp4270 = (data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */);
  tmp4271 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4272 = (tmp4270 * tmp4270) + (tmp4271 * tmp4271);
  tmp4273 = 1.5;
  if(tmp4272 < 0.0 && tmp4273 != 0.0)
  {
    tmp4275 = modf(tmp4273, &tmp4276);
    
    if(tmp4275 > 0.5)
    {
      tmp4275 -= 1.0;
      tmp4276 += 1.0;
    }
    else if(tmp4275 < -0.5)
    {
      tmp4275 += 1.0;
      tmp4276 -= 1.0;
    }
    
    if(fabs(tmp4275) < 1e-10)
      tmp4274 = pow(tmp4272, tmp4276);
    else
    {
      tmp4278 = modf(1.0/tmp4273, &tmp4277);
      if(tmp4278 > 0.5)
      {
        tmp4278 -= 1.0;
        tmp4277 += 1.0;
      }
      else if(tmp4278 < -0.5)
      {
        tmp4278 += 1.0;
        tmp4277 -= 1.0;
      }
      if(fabs(tmp4278) < 1e-10 && ((unsigned long)tmp4277 & 1))
      {
        tmp4274 = -pow(-tmp4272, tmp4275)*pow(tmp4272, tmp4276);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4272, tmp4273);
      }
    }
  }
  else
  {
    tmp4274 = pow(tmp4272, tmp4273);
  }
  if(isnan(tmp4274) || isinf(tmp4274))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4272, tmp4273);
  }tmp4279 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4274,"(r_init[73] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4279 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[73] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4279);
    }
  }
  (data->simulationInfo->realParameter[577] /* omega_c[73] PARAM */) = sqrt(tmp4279);
  TRACE_POP
}

/*
equation index: 13857
type: SIMPLE_ASSIGN
r_init[72] = r_min + 72.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13857};
  (data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (72.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13858
type: SIMPLE_ASSIGN
omega_c[72] = sqrt(G * Md / (r_init[72] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13858(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13858};
  modelica_real tmp4280;
  modelica_real tmp4281;
  modelica_real tmp4282;
  modelica_real tmp4283;
  modelica_real tmp4284;
  modelica_real tmp4285;
  modelica_real tmp4286;
  modelica_real tmp4287;
  modelica_real tmp4288;
  modelica_real tmp4289;
  tmp4280 = (data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */);
  tmp4281 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4282 = (tmp4280 * tmp4280) + (tmp4281 * tmp4281);
  tmp4283 = 1.5;
  if(tmp4282 < 0.0 && tmp4283 != 0.0)
  {
    tmp4285 = modf(tmp4283, &tmp4286);
    
    if(tmp4285 > 0.5)
    {
      tmp4285 -= 1.0;
      tmp4286 += 1.0;
    }
    else if(tmp4285 < -0.5)
    {
      tmp4285 += 1.0;
      tmp4286 -= 1.0;
    }
    
    if(fabs(tmp4285) < 1e-10)
      tmp4284 = pow(tmp4282, tmp4286);
    else
    {
      tmp4288 = modf(1.0/tmp4283, &tmp4287);
      if(tmp4288 > 0.5)
      {
        tmp4288 -= 1.0;
        tmp4287 += 1.0;
      }
      else if(tmp4288 < -0.5)
      {
        tmp4288 += 1.0;
        tmp4287 -= 1.0;
      }
      if(fabs(tmp4288) < 1e-10 && ((unsigned long)tmp4287 & 1))
      {
        tmp4284 = -pow(-tmp4282, tmp4285)*pow(tmp4282, tmp4286);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4282, tmp4283);
      }
    }
  }
  else
  {
    tmp4284 = pow(tmp4282, tmp4283);
  }
  if(isnan(tmp4284) || isinf(tmp4284))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4282, tmp4283);
  }tmp4289 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4284,"(r_init[72] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4289 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[72] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4289);
    }
  }
  (data->simulationInfo->realParameter[576] /* omega_c[72] PARAM */) = sqrt(tmp4289);
  TRACE_POP
}

/*
equation index: 13859
type: SIMPLE_ASSIGN
r_init[71] = r_min + 71.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13859};
  (data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (71.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13860
type: SIMPLE_ASSIGN
omega_c[71] = sqrt(G * Md / (r_init[71] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13860};
  modelica_real tmp4290;
  modelica_real tmp4291;
  modelica_real tmp4292;
  modelica_real tmp4293;
  modelica_real tmp4294;
  modelica_real tmp4295;
  modelica_real tmp4296;
  modelica_real tmp4297;
  modelica_real tmp4298;
  modelica_real tmp4299;
  tmp4290 = (data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */);
  tmp4291 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4292 = (tmp4290 * tmp4290) + (tmp4291 * tmp4291);
  tmp4293 = 1.5;
  if(tmp4292 < 0.0 && tmp4293 != 0.0)
  {
    tmp4295 = modf(tmp4293, &tmp4296);
    
    if(tmp4295 > 0.5)
    {
      tmp4295 -= 1.0;
      tmp4296 += 1.0;
    }
    else if(tmp4295 < -0.5)
    {
      tmp4295 += 1.0;
      tmp4296 -= 1.0;
    }
    
    if(fabs(tmp4295) < 1e-10)
      tmp4294 = pow(tmp4292, tmp4296);
    else
    {
      tmp4298 = modf(1.0/tmp4293, &tmp4297);
      if(tmp4298 > 0.5)
      {
        tmp4298 -= 1.0;
        tmp4297 += 1.0;
      }
      else if(tmp4298 < -0.5)
      {
        tmp4298 += 1.0;
        tmp4297 -= 1.0;
      }
      if(fabs(tmp4298) < 1e-10 && ((unsigned long)tmp4297 & 1))
      {
        tmp4294 = -pow(-tmp4292, tmp4295)*pow(tmp4292, tmp4296);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4292, tmp4293);
      }
    }
  }
  else
  {
    tmp4294 = pow(tmp4292, tmp4293);
  }
  if(isnan(tmp4294) || isinf(tmp4294))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4292, tmp4293);
  }tmp4299 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4294,"(r_init[71] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4299 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[71] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4299);
    }
  }
  (data->simulationInfo->realParameter[575] /* omega_c[71] PARAM */) = sqrt(tmp4299);
  TRACE_POP
}

/*
equation index: 13861
type: SIMPLE_ASSIGN
r_init[70] = r_min + 70.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13861};
  (data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (70.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13862
type: SIMPLE_ASSIGN
omega_c[70] = sqrt(G * Md / (r_init[70] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13862(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13862};
  modelica_real tmp4300;
  modelica_real tmp4301;
  modelica_real tmp4302;
  modelica_real tmp4303;
  modelica_real tmp4304;
  modelica_real tmp4305;
  modelica_real tmp4306;
  modelica_real tmp4307;
  modelica_real tmp4308;
  modelica_real tmp4309;
  tmp4300 = (data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */);
  tmp4301 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4302 = (tmp4300 * tmp4300) + (tmp4301 * tmp4301);
  tmp4303 = 1.5;
  if(tmp4302 < 0.0 && tmp4303 != 0.0)
  {
    tmp4305 = modf(tmp4303, &tmp4306);
    
    if(tmp4305 > 0.5)
    {
      tmp4305 -= 1.0;
      tmp4306 += 1.0;
    }
    else if(tmp4305 < -0.5)
    {
      tmp4305 += 1.0;
      tmp4306 -= 1.0;
    }
    
    if(fabs(tmp4305) < 1e-10)
      tmp4304 = pow(tmp4302, tmp4306);
    else
    {
      tmp4308 = modf(1.0/tmp4303, &tmp4307);
      if(tmp4308 > 0.5)
      {
        tmp4308 -= 1.0;
        tmp4307 += 1.0;
      }
      else if(tmp4308 < -0.5)
      {
        tmp4308 += 1.0;
        tmp4307 -= 1.0;
      }
      if(fabs(tmp4308) < 1e-10 && ((unsigned long)tmp4307 & 1))
      {
        tmp4304 = -pow(-tmp4302, tmp4305)*pow(tmp4302, tmp4306);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4302, tmp4303);
      }
    }
  }
  else
  {
    tmp4304 = pow(tmp4302, tmp4303);
  }
  if(isnan(tmp4304) || isinf(tmp4304))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4302, tmp4303);
  }tmp4309 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4304,"(r_init[70] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4309 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[70] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4309);
    }
  }
  (data->simulationInfo->realParameter[574] /* omega_c[70] PARAM */) = sqrt(tmp4309);
  TRACE_POP
}

/*
equation index: 13863
type: SIMPLE_ASSIGN
r_init[69] = r_min + 69.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13863};
  (data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (69.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13864
type: SIMPLE_ASSIGN
omega_c[69] = sqrt(G * Md / (r_init[69] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13864(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13864};
  modelica_real tmp4310;
  modelica_real tmp4311;
  modelica_real tmp4312;
  modelica_real tmp4313;
  modelica_real tmp4314;
  modelica_real tmp4315;
  modelica_real tmp4316;
  modelica_real tmp4317;
  modelica_real tmp4318;
  modelica_real tmp4319;
  tmp4310 = (data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */);
  tmp4311 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4312 = (tmp4310 * tmp4310) + (tmp4311 * tmp4311);
  tmp4313 = 1.5;
  if(tmp4312 < 0.0 && tmp4313 != 0.0)
  {
    tmp4315 = modf(tmp4313, &tmp4316);
    
    if(tmp4315 > 0.5)
    {
      tmp4315 -= 1.0;
      tmp4316 += 1.0;
    }
    else if(tmp4315 < -0.5)
    {
      tmp4315 += 1.0;
      tmp4316 -= 1.0;
    }
    
    if(fabs(tmp4315) < 1e-10)
      tmp4314 = pow(tmp4312, tmp4316);
    else
    {
      tmp4318 = modf(1.0/tmp4313, &tmp4317);
      if(tmp4318 > 0.5)
      {
        tmp4318 -= 1.0;
        tmp4317 += 1.0;
      }
      else if(tmp4318 < -0.5)
      {
        tmp4318 += 1.0;
        tmp4317 -= 1.0;
      }
      if(fabs(tmp4318) < 1e-10 && ((unsigned long)tmp4317 & 1))
      {
        tmp4314 = -pow(-tmp4312, tmp4315)*pow(tmp4312, tmp4316);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4312, tmp4313);
      }
    }
  }
  else
  {
    tmp4314 = pow(tmp4312, tmp4313);
  }
  if(isnan(tmp4314) || isinf(tmp4314))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4312, tmp4313);
  }tmp4319 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4314,"(r_init[69] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4319 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[69] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4319);
    }
  }
  (data->simulationInfo->realParameter[573] /* omega_c[69] PARAM */) = sqrt(tmp4319);
  TRACE_POP
}

/*
equation index: 13865
type: SIMPLE_ASSIGN
r_init[68] = r_min + 68.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13865};
  (data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (68.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13866
type: SIMPLE_ASSIGN
omega_c[68] = sqrt(G * Md / (r_init[68] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13866(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13866};
  modelica_real tmp4320;
  modelica_real tmp4321;
  modelica_real tmp4322;
  modelica_real tmp4323;
  modelica_real tmp4324;
  modelica_real tmp4325;
  modelica_real tmp4326;
  modelica_real tmp4327;
  modelica_real tmp4328;
  modelica_real tmp4329;
  tmp4320 = (data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */);
  tmp4321 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4322 = (tmp4320 * tmp4320) + (tmp4321 * tmp4321);
  tmp4323 = 1.5;
  if(tmp4322 < 0.0 && tmp4323 != 0.0)
  {
    tmp4325 = modf(tmp4323, &tmp4326);
    
    if(tmp4325 > 0.5)
    {
      tmp4325 -= 1.0;
      tmp4326 += 1.0;
    }
    else if(tmp4325 < -0.5)
    {
      tmp4325 += 1.0;
      tmp4326 -= 1.0;
    }
    
    if(fabs(tmp4325) < 1e-10)
      tmp4324 = pow(tmp4322, tmp4326);
    else
    {
      tmp4328 = modf(1.0/tmp4323, &tmp4327);
      if(tmp4328 > 0.5)
      {
        tmp4328 -= 1.0;
        tmp4327 += 1.0;
      }
      else if(tmp4328 < -0.5)
      {
        tmp4328 += 1.0;
        tmp4327 -= 1.0;
      }
      if(fabs(tmp4328) < 1e-10 && ((unsigned long)tmp4327 & 1))
      {
        tmp4324 = -pow(-tmp4322, tmp4325)*pow(tmp4322, tmp4326);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4322, tmp4323);
      }
    }
  }
  else
  {
    tmp4324 = pow(tmp4322, tmp4323);
  }
  if(isnan(tmp4324) || isinf(tmp4324))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4322, tmp4323);
  }tmp4329 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4324,"(r_init[68] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4329 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[68] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4329);
    }
  }
  (data->simulationInfo->realParameter[572] /* omega_c[68] PARAM */) = sqrt(tmp4329);
  TRACE_POP
}

/*
equation index: 13867
type: SIMPLE_ASSIGN
r_init[67] = r_min + 67.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13867};
  (data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (67.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13868
type: SIMPLE_ASSIGN
omega_c[67] = sqrt(G * Md / (r_init[67] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13868};
  modelica_real tmp4330;
  modelica_real tmp4331;
  modelica_real tmp4332;
  modelica_real tmp4333;
  modelica_real tmp4334;
  modelica_real tmp4335;
  modelica_real tmp4336;
  modelica_real tmp4337;
  modelica_real tmp4338;
  modelica_real tmp4339;
  tmp4330 = (data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */);
  tmp4331 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4332 = (tmp4330 * tmp4330) + (tmp4331 * tmp4331);
  tmp4333 = 1.5;
  if(tmp4332 < 0.0 && tmp4333 != 0.0)
  {
    tmp4335 = modf(tmp4333, &tmp4336);
    
    if(tmp4335 > 0.5)
    {
      tmp4335 -= 1.0;
      tmp4336 += 1.0;
    }
    else if(tmp4335 < -0.5)
    {
      tmp4335 += 1.0;
      tmp4336 -= 1.0;
    }
    
    if(fabs(tmp4335) < 1e-10)
      tmp4334 = pow(tmp4332, tmp4336);
    else
    {
      tmp4338 = modf(1.0/tmp4333, &tmp4337);
      if(tmp4338 > 0.5)
      {
        tmp4338 -= 1.0;
        tmp4337 += 1.0;
      }
      else if(tmp4338 < -0.5)
      {
        tmp4338 += 1.0;
        tmp4337 -= 1.0;
      }
      if(fabs(tmp4338) < 1e-10 && ((unsigned long)tmp4337 & 1))
      {
        tmp4334 = -pow(-tmp4332, tmp4335)*pow(tmp4332, tmp4336);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4332, tmp4333);
      }
    }
  }
  else
  {
    tmp4334 = pow(tmp4332, tmp4333);
  }
  if(isnan(tmp4334) || isinf(tmp4334))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4332, tmp4333);
  }tmp4339 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4334,"(r_init[67] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4339 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[67] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4339);
    }
  }
  (data->simulationInfo->realParameter[571] /* omega_c[67] PARAM */) = sqrt(tmp4339);
  TRACE_POP
}

/*
equation index: 13869
type: SIMPLE_ASSIGN
r_init[66] = r_min + 66.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13869};
  (data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (66.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13870
type: SIMPLE_ASSIGN
omega_c[66] = sqrt(G * Md / (r_init[66] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13870(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13870};
  modelica_real tmp4340;
  modelica_real tmp4341;
  modelica_real tmp4342;
  modelica_real tmp4343;
  modelica_real tmp4344;
  modelica_real tmp4345;
  modelica_real tmp4346;
  modelica_real tmp4347;
  modelica_real tmp4348;
  modelica_real tmp4349;
  tmp4340 = (data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */);
  tmp4341 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4342 = (tmp4340 * tmp4340) + (tmp4341 * tmp4341);
  tmp4343 = 1.5;
  if(tmp4342 < 0.0 && tmp4343 != 0.0)
  {
    tmp4345 = modf(tmp4343, &tmp4346);
    
    if(tmp4345 > 0.5)
    {
      tmp4345 -= 1.0;
      tmp4346 += 1.0;
    }
    else if(tmp4345 < -0.5)
    {
      tmp4345 += 1.0;
      tmp4346 -= 1.0;
    }
    
    if(fabs(tmp4345) < 1e-10)
      tmp4344 = pow(tmp4342, tmp4346);
    else
    {
      tmp4348 = modf(1.0/tmp4343, &tmp4347);
      if(tmp4348 > 0.5)
      {
        tmp4348 -= 1.0;
        tmp4347 += 1.0;
      }
      else if(tmp4348 < -0.5)
      {
        tmp4348 += 1.0;
        tmp4347 -= 1.0;
      }
      if(fabs(tmp4348) < 1e-10 && ((unsigned long)tmp4347 & 1))
      {
        tmp4344 = -pow(-tmp4342, tmp4345)*pow(tmp4342, tmp4346);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4342, tmp4343);
      }
    }
  }
  else
  {
    tmp4344 = pow(tmp4342, tmp4343);
  }
  if(isnan(tmp4344) || isinf(tmp4344))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4342, tmp4343);
  }tmp4349 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4344,"(r_init[66] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4349 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[66] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4349);
    }
  }
  (data->simulationInfo->realParameter[570] /* omega_c[66] PARAM */) = sqrt(tmp4349);
  TRACE_POP
}

/*
equation index: 13871
type: SIMPLE_ASSIGN
r_init[65] = r_min + 65.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13871};
  (data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (65.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13872
type: SIMPLE_ASSIGN
omega_c[65] = sqrt(G * Md / (r_init[65] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13872(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13872};
  modelica_real tmp4350;
  modelica_real tmp4351;
  modelica_real tmp4352;
  modelica_real tmp4353;
  modelica_real tmp4354;
  modelica_real tmp4355;
  modelica_real tmp4356;
  modelica_real tmp4357;
  modelica_real tmp4358;
  modelica_real tmp4359;
  tmp4350 = (data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */);
  tmp4351 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4352 = (tmp4350 * tmp4350) + (tmp4351 * tmp4351);
  tmp4353 = 1.5;
  if(tmp4352 < 0.0 && tmp4353 != 0.0)
  {
    tmp4355 = modf(tmp4353, &tmp4356);
    
    if(tmp4355 > 0.5)
    {
      tmp4355 -= 1.0;
      tmp4356 += 1.0;
    }
    else if(tmp4355 < -0.5)
    {
      tmp4355 += 1.0;
      tmp4356 -= 1.0;
    }
    
    if(fabs(tmp4355) < 1e-10)
      tmp4354 = pow(tmp4352, tmp4356);
    else
    {
      tmp4358 = modf(1.0/tmp4353, &tmp4357);
      if(tmp4358 > 0.5)
      {
        tmp4358 -= 1.0;
        tmp4357 += 1.0;
      }
      else if(tmp4358 < -0.5)
      {
        tmp4358 += 1.0;
        tmp4357 -= 1.0;
      }
      if(fabs(tmp4358) < 1e-10 && ((unsigned long)tmp4357 & 1))
      {
        tmp4354 = -pow(-tmp4352, tmp4355)*pow(tmp4352, tmp4356);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4352, tmp4353);
      }
    }
  }
  else
  {
    tmp4354 = pow(tmp4352, tmp4353);
  }
  if(isnan(tmp4354) || isinf(tmp4354))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4352, tmp4353);
  }tmp4359 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4354,"(r_init[65] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4359 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[65] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4359);
    }
  }
  (data->simulationInfo->realParameter[569] /* omega_c[65] PARAM */) = sqrt(tmp4359);
  TRACE_POP
}

/*
equation index: 13873
type: SIMPLE_ASSIGN
r_init[64] = r_min + 64.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13873};
  (data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (64.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13874
type: SIMPLE_ASSIGN
omega_c[64] = sqrt(G * Md / (r_init[64] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13874(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13874};
  modelica_real tmp4360;
  modelica_real tmp4361;
  modelica_real tmp4362;
  modelica_real tmp4363;
  modelica_real tmp4364;
  modelica_real tmp4365;
  modelica_real tmp4366;
  modelica_real tmp4367;
  modelica_real tmp4368;
  modelica_real tmp4369;
  tmp4360 = (data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */);
  tmp4361 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4362 = (tmp4360 * tmp4360) + (tmp4361 * tmp4361);
  tmp4363 = 1.5;
  if(tmp4362 < 0.0 && tmp4363 != 0.0)
  {
    tmp4365 = modf(tmp4363, &tmp4366);
    
    if(tmp4365 > 0.5)
    {
      tmp4365 -= 1.0;
      tmp4366 += 1.0;
    }
    else if(tmp4365 < -0.5)
    {
      tmp4365 += 1.0;
      tmp4366 -= 1.0;
    }
    
    if(fabs(tmp4365) < 1e-10)
      tmp4364 = pow(tmp4362, tmp4366);
    else
    {
      tmp4368 = modf(1.0/tmp4363, &tmp4367);
      if(tmp4368 > 0.5)
      {
        tmp4368 -= 1.0;
        tmp4367 += 1.0;
      }
      else if(tmp4368 < -0.5)
      {
        tmp4368 += 1.0;
        tmp4367 -= 1.0;
      }
      if(fabs(tmp4368) < 1e-10 && ((unsigned long)tmp4367 & 1))
      {
        tmp4364 = -pow(-tmp4362, tmp4365)*pow(tmp4362, tmp4366);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4362, tmp4363);
      }
    }
  }
  else
  {
    tmp4364 = pow(tmp4362, tmp4363);
  }
  if(isnan(tmp4364) || isinf(tmp4364))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4362, tmp4363);
  }tmp4369 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4364,"(r_init[64] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4369 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[64] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4369);
    }
  }
  (data->simulationInfo->realParameter[568] /* omega_c[64] PARAM */) = sqrt(tmp4369);
  TRACE_POP
}

/*
equation index: 13875
type: SIMPLE_ASSIGN
r_init[63] = r_min + 63.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13875};
  (data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (63.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13876
type: SIMPLE_ASSIGN
omega_c[63] = sqrt(G * Md / (r_init[63] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13876};
  modelica_real tmp4370;
  modelica_real tmp4371;
  modelica_real tmp4372;
  modelica_real tmp4373;
  modelica_real tmp4374;
  modelica_real tmp4375;
  modelica_real tmp4376;
  modelica_real tmp4377;
  modelica_real tmp4378;
  modelica_real tmp4379;
  tmp4370 = (data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */);
  tmp4371 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4372 = (tmp4370 * tmp4370) + (tmp4371 * tmp4371);
  tmp4373 = 1.5;
  if(tmp4372 < 0.0 && tmp4373 != 0.0)
  {
    tmp4375 = modf(tmp4373, &tmp4376);
    
    if(tmp4375 > 0.5)
    {
      tmp4375 -= 1.0;
      tmp4376 += 1.0;
    }
    else if(tmp4375 < -0.5)
    {
      tmp4375 += 1.0;
      tmp4376 -= 1.0;
    }
    
    if(fabs(tmp4375) < 1e-10)
      tmp4374 = pow(tmp4372, tmp4376);
    else
    {
      tmp4378 = modf(1.0/tmp4373, &tmp4377);
      if(tmp4378 > 0.5)
      {
        tmp4378 -= 1.0;
        tmp4377 += 1.0;
      }
      else if(tmp4378 < -0.5)
      {
        tmp4378 += 1.0;
        tmp4377 -= 1.0;
      }
      if(fabs(tmp4378) < 1e-10 && ((unsigned long)tmp4377 & 1))
      {
        tmp4374 = -pow(-tmp4372, tmp4375)*pow(tmp4372, tmp4376);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4372, tmp4373);
      }
    }
  }
  else
  {
    tmp4374 = pow(tmp4372, tmp4373);
  }
  if(isnan(tmp4374) || isinf(tmp4374))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4372, tmp4373);
  }tmp4379 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4374,"(r_init[63] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4379 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[63] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4379);
    }
  }
  (data->simulationInfo->realParameter[567] /* omega_c[63] PARAM */) = sqrt(tmp4379);
  TRACE_POP
}

/*
equation index: 13877
type: SIMPLE_ASSIGN
r_init[62] = r_min + 62.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13877};
  (data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (62.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13878
type: SIMPLE_ASSIGN
omega_c[62] = sqrt(G * Md / (r_init[62] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13878(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13878};
  modelica_real tmp4380;
  modelica_real tmp4381;
  modelica_real tmp4382;
  modelica_real tmp4383;
  modelica_real tmp4384;
  modelica_real tmp4385;
  modelica_real tmp4386;
  modelica_real tmp4387;
  modelica_real tmp4388;
  modelica_real tmp4389;
  tmp4380 = (data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */);
  tmp4381 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4382 = (tmp4380 * tmp4380) + (tmp4381 * tmp4381);
  tmp4383 = 1.5;
  if(tmp4382 < 0.0 && tmp4383 != 0.0)
  {
    tmp4385 = modf(tmp4383, &tmp4386);
    
    if(tmp4385 > 0.5)
    {
      tmp4385 -= 1.0;
      tmp4386 += 1.0;
    }
    else if(tmp4385 < -0.5)
    {
      tmp4385 += 1.0;
      tmp4386 -= 1.0;
    }
    
    if(fabs(tmp4385) < 1e-10)
      tmp4384 = pow(tmp4382, tmp4386);
    else
    {
      tmp4388 = modf(1.0/tmp4383, &tmp4387);
      if(tmp4388 > 0.5)
      {
        tmp4388 -= 1.0;
        tmp4387 += 1.0;
      }
      else if(tmp4388 < -0.5)
      {
        tmp4388 += 1.0;
        tmp4387 -= 1.0;
      }
      if(fabs(tmp4388) < 1e-10 && ((unsigned long)tmp4387 & 1))
      {
        tmp4384 = -pow(-tmp4382, tmp4385)*pow(tmp4382, tmp4386);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4382, tmp4383);
      }
    }
  }
  else
  {
    tmp4384 = pow(tmp4382, tmp4383);
  }
  if(isnan(tmp4384) || isinf(tmp4384))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4382, tmp4383);
  }tmp4389 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4384,"(r_init[62] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4389 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[62] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4389);
    }
  }
  (data->simulationInfo->realParameter[566] /* omega_c[62] PARAM */) = sqrt(tmp4389);
  TRACE_POP
}

/*
equation index: 13879
type: SIMPLE_ASSIGN
r_init[61] = r_min + 61.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13879};
  (data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (61.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13880
type: SIMPLE_ASSIGN
omega_c[61] = sqrt(G * Md / (r_init[61] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13880(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13880};
  modelica_real tmp4390;
  modelica_real tmp4391;
  modelica_real tmp4392;
  modelica_real tmp4393;
  modelica_real tmp4394;
  modelica_real tmp4395;
  modelica_real tmp4396;
  modelica_real tmp4397;
  modelica_real tmp4398;
  modelica_real tmp4399;
  tmp4390 = (data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */);
  tmp4391 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4392 = (tmp4390 * tmp4390) + (tmp4391 * tmp4391);
  tmp4393 = 1.5;
  if(tmp4392 < 0.0 && tmp4393 != 0.0)
  {
    tmp4395 = modf(tmp4393, &tmp4396);
    
    if(tmp4395 > 0.5)
    {
      tmp4395 -= 1.0;
      tmp4396 += 1.0;
    }
    else if(tmp4395 < -0.5)
    {
      tmp4395 += 1.0;
      tmp4396 -= 1.0;
    }
    
    if(fabs(tmp4395) < 1e-10)
      tmp4394 = pow(tmp4392, tmp4396);
    else
    {
      tmp4398 = modf(1.0/tmp4393, &tmp4397);
      if(tmp4398 > 0.5)
      {
        tmp4398 -= 1.0;
        tmp4397 += 1.0;
      }
      else if(tmp4398 < -0.5)
      {
        tmp4398 += 1.0;
        tmp4397 -= 1.0;
      }
      if(fabs(tmp4398) < 1e-10 && ((unsigned long)tmp4397 & 1))
      {
        tmp4394 = -pow(-tmp4392, tmp4395)*pow(tmp4392, tmp4396);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4392, tmp4393);
      }
    }
  }
  else
  {
    tmp4394 = pow(tmp4392, tmp4393);
  }
  if(isnan(tmp4394) || isinf(tmp4394))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4392, tmp4393);
  }tmp4399 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4394,"(r_init[61] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4399 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[61] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4399);
    }
  }
  (data->simulationInfo->realParameter[565] /* omega_c[61] PARAM */) = sqrt(tmp4399);
  TRACE_POP
}

/*
equation index: 13881
type: SIMPLE_ASSIGN
r_init[60] = r_min + 60.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13881};
  (data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (60.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13882
type: SIMPLE_ASSIGN
omega_c[60] = sqrt(G * Md / (r_init[60] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13882(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13882};
  modelica_real tmp4400;
  modelica_real tmp4401;
  modelica_real tmp4402;
  modelica_real tmp4403;
  modelica_real tmp4404;
  modelica_real tmp4405;
  modelica_real tmp4406;
  modelica_real tmp4407;
  modelica_real tmp4408;
  modelica_real tmp4409;
  tmp4400 = (data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */);
  tmp4401 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4402 = (tmp4400 * tmp4400) + (tmp4401 * tmp4401);
  tmp4403 = 1.5;
  if(tmp4402 < 0.0 && tmp4403 != 0.0)
  {
    tmp4405 = modf(tmp4403, &tmp4406);
    
    if(tmp4405 > 0.5)
    {
      tmp4405 -= 1.0;
      tmp4406 += 1.0;
    }
    else if(tmp4405 < -0.5)
    {
      tmp4405 += 1.0;
      tmp4406 -= 1.0;
    }
    
    if(fabs(tmp4405) < 1e-10)
      tmp4404 = pow(tmp4402, tmp4406);
    else
    {
      tmp4408 = modf(1.0/tmp4403, &tmp4407);
      if(tmp4408 > 0.5)
      {
        tmp4408 -= 1.0;
        tmp4407 += 1.0;
      }
      else if(tmp4408 < -0.5)
      {
        tmp4408 += 1.0;
        tmp4407 -= 1.0;
      }
      if(fabs(tmp4408) < 1e-10 && ((unsigned long)tmp4407 & 1))
      {
        tmp4404 = -pow(-tmp4402, tmp4405)*pow(tmp4402, tmp4406);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4402, tmp4403);
      }
    }
  }
  else
  {
    tmp4404 = pow(tmp4402, tmp4403);
  }
  if(isnan(tmp4404) || isinf(tmp4404))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4402, tmp4403);
  }tmp4409 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4404,"(r_init[60] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4409 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[60] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4409);
    }
  }
  (data->simulationInfo->realParameter[564] /* omega_c[60] PARAM */) = sqrt(tmp4409);
  TRACE_POP
}

/*
equation index: 13883
type: SIMPLE_ASSIGN
r_init[59] = r_min + 59.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13883};
  (data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (59.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13884
type: SIMPLE_ASSIGN
omega_c[59] = sqrt(G * Md / (r_init[59] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13884};
  modelica_real tmp4410;
  modelica_real tmp4411;
  modelica_real tmp4412;
  modelica_real tmp4413;
  modelica_real tmp4414;
  modelica_real tmp4415;
  modelica_real tmp4416;
  modelica_real tmp4417;
  modelica_real tmp4418;
  modelica_real tmp4419;
  tmp4410 = (data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */);
  tmp4411 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4412 = (tmp4410 * tmp4410) + (tmp4411 * tmp4411);
  tmp4413 = 1.5;
  if(tmp4412 < 0.0 && tmp4413 != 0.0)
  {
    tmp4415 = modf(tmp4413, &tmp4416);
    
    if(tmp4415 > 0.5)
    {
      tmp4415 -= 1.0;
      tmp4416 += 1.0;
    }
    else if(tmp4415 < -0.5)
    {
      tmp4415 += 1.0;
      tmp4416 -= 1.0;
    }
    
    if(fabs(tmp4415) < 1e-10)
      tmp4414 = pow(tmp4412, tmp4416);
    else
    {
      tmp4418 = modf(1.0/tmp4413, &tmp4417);
      if(tmp4418 > 0.5)
      {
        tmp4418 -= 1.0;
        tmp4417 += 1.0;
      }
      else if(tmp4418 < -0.5)
      {
        tmp4418 += 1.0;
        tmp4417 -= 1.0;
      }
      if(fabs(tmp4418) < 1e-10 && ((unsigned long)tmp4417 & 1))
      {
        tmp4414 = -pow(-tmp4412, tmp4415)*pow(tmp4412, tmp4416);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4412, tmp4413);
      }
    }
  }
  else
  {
    tmp4414 = pow(tmp4412, tmp4413);
  }
  if(isnan(tmp4414) || isinf(tmp4414))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4412, tmp4413);
  }tmp4419 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4414,"(r_init[59] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4419 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[59] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4419);
    }
  }
  (data->simulationInfo->realParameter[563] /* omega_c[59] PARAM */) = sqrt(tmp4419);
  TRACE_POP
}

/*
equation index: 13885
type: SIMPLE_ASSIGN
r_init[58] = r_min + 58.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13885};
  (data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (58.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13886
type: SIMPLE_ASSIGN
omega_c[58] = sqrt(G * Md / (r_init[58] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13886(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13886};
  modelica_real tmp4420;
  modelica_real tmp4421;
  modelica_real tmp4422;
  modelica_real tmp4423;
  modelica_real tmp4424;
  modelica_real tmp4425;
  modelica_real tmp4426;
  modelica_real tmp4427;
  modelica_real tmp4428;
  modelica_real tmp4429;
  tmp4420 = (data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */);
  tmp4421 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4422 = (tmp4420 * tmp4420) + (tmp4421 * tmp4421);
  tmp4423 = 1.5;
  if(tmp4422 < 0.0 && tmp4423 != 0.0)
  {
    tmp4425 = modf(tmp4423, &tmp4426);
    
    if(tmp4425 > 0.5)
    {
      tmp4425 -= 1.0;
      tmp4426 += 1.0;
    }
    else if(tmp4425 < -0.5)
    {
      tmp4425 += 1.0;
      tmp4426 -= 1.0;
    }
    
    if(fabs(tmp4425) < 1e-10)
      tmp4424 = pow(tmp4422, tmp4426);
    else
    {
      tmp4428 = modf(1.0/tmp4423, &tmp4427);
      if(tmp4428 > 0.5)
      {
        tmp4428 -= 1.0;
        tmp4427 += 1.0;
      }
      else if(tmp4428 < -0.5)
      {
        tmp4428 += 1.0;
        tmp4427 -= 1.0;
      }
      if(fabs(tmp4428) < 1e-10 && ((unsigned long)tmp4427 & 1))
      {
        tmp4424 = -pow(-tmp4422, tmp4425)*pow(tmp4422, tmp4426);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4422, tmp4423);
      }
    }
  }
  else
  {
    tmp4424 = pow(tmp4422, tmp4423);
  }
  if(isnan(tmp4424) || isinf(tmp4424))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4422, tmp4423);
  }tmp4429 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4424,"(r_init[58] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4429 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[58] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4429);
    }
  }
  (data->simulationInfo->realParameter[562] /* omega_c[58] PARAM */) = sqrt(tmp4429);
  TRACE_POP
}

/*
equation index: 13887
type: SIMPLE_ASSIGN
r_init[57] = r_min + 57.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13887};
  (data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (57.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13888
type: SIMPLE_ASSIGN
omega_c[57] = sqrt(G * Md / (r_init[57] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13888(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13888};
  modelica_real tmp4430;
  modelica_real tmp4431;
  modelica_real tmp4432;
  modelica_real tmp4433;
  modelica_real tmp4434;
  modelica_real tmp4435;
  modelica_real tmp4436;
  modelica_real tmp4437;
  modelica_real tmp4438;
  modelica_real tmp4439;
  tmp4430 = (data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */);
  tmp4431 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4432 = (tmp4430 * tmp4430) + (tmp4431 * tmp4431);
  tmp4433 = 1.5;
  if(tmp4432 < 0.0 && tmp4433 != 0.0)
  {
    tmp4435 = modf(tmp4433, &tmp4436);
    
    if(tmp4435 > 0.5)
    {
      tmp4435 -= 1.0;
      tmp4436 += 1.0;
    }
    else if(tmp4435 < -0.5)
    {
      tmp4435 += 1.0;
      tmp4436 -= 1.0;
    }
    
    if(fabs(tmp4435) < 1e-10)
      tmp4434 = pow(tmp4432, tmp4436);
    else
    {
      tmp4438 = modf(1.0/tmp4433, &tmp4437);
      if(tmp4438 > 0.5)
      {
        tmp4438 -= 1.0;
        tmp4437 += 1.0;
      }
      else if(tmp4438 < -0.5)
      {
        tmp4438 += 1.0;
        tmp4437 -= 1.0;
      }
      if(fabs(tmp4438) < 1e-10 && ((unsigned long)tmp4437 & 1))
      {
        tmp4434 = -pow(-tmp4432, tmp4435)*pow(tmp4432, tmp4436);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4432, tmp4433);
      }
    }
  }
  else
  {
    tmp4434 = pow(tmp4432, tmp4433);
  }
  if(isnan(tmp4434) || isinf(tmp4434))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4432, tmp4433);
  }tmp4439 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4434,"(r_init[57] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4439 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[57] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4439);
    }
  }
  (data->simulationInfo->realParameter[561] /* omega_c[57] PARAM */) = sqrt(tmp4439);
  TRACE_POP
}

/*
equation index: 13889
type: SIMPLE_ASSIGN
r_init[56] = r_min + 56.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13889};
  (data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (56.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13890
type: SIMPLE_ASSIGN
omega_c[56] = sqrt(G * Md / (r_init[56] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13890(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13890};
  modelica_real tmp4440;
  modelica_real tmp4441;
  modelica_real tmp4442;
  modelica_real tmp4443;
  modelica_real tmp4444;
  modelica_real tmp4445;
  modelica_real tmp4446;
  modelica_real tmp4447;
  modelica_real tmp4448;
  modelica_real tmp4449;
  tmp4440 = (data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */);
  tmp4441 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4442 = (tmp4440 * tmp4440) + (tmp4441 * tmp4441);
  tmp4443 = 1.5;
  if(tmp4442 < 0.0 && tmp4443 != 0.0)
  {
    tmp4445 = modf(tmp4443, &tmp4446);
    
    if(tmp4445 > 0.5)
    {
      tmp4445 -= 1.0;
      tmp4446 += 1.0;
    }
    else if(tmp4445 < -0.5)
    {
      tmp4445 += 1.0;
      tmp4446 -= 1.0;
    }
    
    if(fabs(tmp4445) < 1e-10)
      tmp4444 = pow(tmp4442, tmp4446);
    else
    {
      tmp4448 = modf(1.0/tmp4443, &tmp4447);
      if(tmp4448 > 0.5)
      {
        tmp4448 -= 1.0;
        tmp4447 += 1.0;
      }
      else if(tmp4448 < -0.5)
      {
        tmp4448 += 1.0;
        tmp4447 -= 1.0;
      }
      if(fabs(tmp4448) < 1e-10 && ((unsigned long)tmp4447 & 1))
      {
        tmp4444 = -pow(-tmp4442, tmp4445)*pow(tmp4442, tmp4446);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4442, tmp4443);
      }
    }
  }
  else
  {
    tmp4444 = pow(tmp4442, tmp4443);
  }
  if(isnan(tmp4444) || isinf(tmp4444))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4442, tmp4443);
  }tmp4449 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4444,"(r_init[56] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4449 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[56] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4449);
    }
  }
  (data->simulationInfo->realParameter[560] /* omega_c[56] PARAM */) = sqrt(tmp4449);
  TRACE_POP
}

/*
equation index: 13891
type: SIMPLE_ASSIGN
r_init[55] = r_min + 55.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13891};
  (data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (55.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13892
type: SIMPLE_ASSIGN
omega_c[55] = sqrt(G * Md / (r_init[55] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13892};
  modelica_real tmp4450;
  modelica_real tmp4451;
  modelica_real tmp4452;
  modelica_real tmp4453;
  modelica_real tmp4454;
  modelica_real tmp4455;
  modelica_real tmp4456;
  modelica_real tmp4457;
  modelica_real tmp4458;
  modelica_real tmp4459;
  tmp4450 = (data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */);
  tmp4451 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4452 = (tmp4450 * tmp4450) + (tmp4451 * tmp4451);
  tmp4453 = 1.5;
  if(tmp4452 < 0.0 && tmp4453 != 0.0)
  {
    tmp4455 = modf(tmp4453, &tmp4456);
    
    if(tmp4455 > 0.5)
    {
      tmp4455 -= 1.0;
      tmp4456 += 1.0;
    }
    else if(tmp4455 < -0.5)
    {
      tmp4455 += 1.0;
      tmp4456 -= 1.0;
    }
    
    if(fabs(tmp4455) < 1e-10)
      tmp4454 = pow(tmp4452, tmp4456);
    else
    {
      tmp4458 = modf(1.0/tmp4453, &tmp4457);
      if(tmp4458 > 0.5)
      {
        tmp4458 -= 1.0;
        tmp4457 += 1.0;
      }
      else if(tmp4458 < -0.5)
      {
        tmp4458 += 1.0;
        tmp4457 -= 1.0;
      }
      if(fabs(tmp4458) < 1e-10 && ((unsigned long)tmp4457 & 1))
      {
        tmp4454 = -pow(-tmp4452, tmp4455)*pow(tmp4452, tmp4456);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4452, tmp4453);
      }
    }
  }
  else
  {
    tmp4454 = pow(tmp4452, tmp4453);
  }
  if(isnan(tmp4454) || isinf(tmp4454))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4452, tmp4453);
  }tmp4459 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4454,"(r_init[55] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4459 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[55] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4459);
    }
  }
  (data->simulationInfo->realParameter[559] /* omega_c[55] PARAM */) = sqrt(tmp4459);
  TRACE_POP
}

/*
equation index: 13893
type: SIMPLE_ASSIGN
r_init[54] = r_min + 54.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13893};
  (data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (54.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13894
type: SIMPLE_ASSIGN
omega_c[54] = sqrt(G * Md / (r_init[54] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13894(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13894};
  modelica_real tmp4460;
  modelica_real tmp4461;
  modelica_real tmp4462;
  modelica_real tmp4463;
  modelica_real tmp4464;
  modelica_real tmp4465;
  modelica_real tmp4466;
  modelica_real tmp4467;
  modelica_real tmp4468;
  modelica_real tmp4469;
  tmp4460 = (data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */);
  tmp4461 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4462 = (tmp4460 * tmp4460) + (tmp4461 * tmp4461);
  tmp4463 = 1.5;
  if(tmp4462 < 0.0 && tmp4463 != 0.0)
  {
    tmp4465 = modf(tmp4463, &tmp4466);
    
    if(tmp4465 > 0.5)
    {
      tmp4465 -= 1.0;
      tmp4466 += 1.0;
    }
    else if(tmp4465 < -0.5)
    {
      tmp4465 += 1.0;
      tmp4466 -= 1.0;
    }
    
    if(fabs(tmp4465) < 1e-10)
      tmp4464 = pow(tmp4462, tmp4466);
    else
    {
      tmp4468 = modf(1.0/tmp4463, &tmp4467);
      if(tmp4468 > 0.5)
      {
        tmp4468 -= 1.0;
        tmp4467 += 1.0;
      }
      else if(tmp4468 < -0.5)
      {
        tmp4468 += 1.0;
        tmp4467 -= 1.0;
      }
      if(fabs(tmp4468) < 1e-10 && ((unsigned long)tmp4467 & 1))
      {
        tmp4464 = -pow(-tmp4462, tmp4465)*pow(tmp4462, tmp4466);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4462, tmp4463);
      }
    }
  }
  else
  {
    tmp4464 = pow(tmp4462, tmp4463);
  }
  if(isnan(tmp4464) || isinf(tmp4464))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4462, tmp4463);
  }tmp4469 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4464,"(r_init[54] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4469 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[54] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4469);
    }
  }
  (data->simulationInfo->realParameter[558] /* omega_c[54] PARAM */) = sqrt(tmp4469);
  TRACE_POP
}

/*
equation index: 13895
type: SIMPLE_ASSIGN
r_init[53] = r_min + 53.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13895};
  (data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (53.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13896
type: SIMPLE_ASSIGN
omega_c[53] = sqrt(G * Md / (r_init[53] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13896(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13896};
  modelica_real tmp4470;
  modelica_real tmp4471;
  modelica_real tmp4472;
  modelica_real tmp4473;
  modelica_real tmp4474;
  modelica_real tmp4475;
  modelica_real tmp4476;
  modelica_real tmp4477;
  modelica_real tmp4478;
  modelica_real tmp4479;
  tmp4470 = (data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */);
  tmp4471 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4472 = (tmp4470 * tmp4470) + (tmp4471 * tmp4471);
  tmp4473 = 1.5;
  if(tmp4472 < 0.0 && tmp4473 != 0.0)
  {
    tmp4475 = modf(tmp4473, &tmp4476);
    
    if(tmp4475 > 0.5)
    {
      tmp4475 -= 1.0;
      tmp4476 += 1.0;
    }
    else if(tmp4475 < -0.5)
    {
      tmp4475 += 1.0;
      tmp4476 -= 1.0;
    }
    
    if(fabs(tmp4475) < 1e-10)
      tmp4474 = pow(tmp4472, tmp4476);
    else
    {
      tmp4478 = modf(1.0/tmp4473, &tmp4477);
      if(tmp4478 > 0.5)
      {
        tmp4478 -= 1.0;
        tmp4477 += 1.0;
      }
      else if(tmp4478 < -0.5)
      {
        tmp4478 += 1.0;
        tmp4477 -= 1.0;
      }
      if(fabs(tmp4478) < 1e-10 && ((unsigned long)tmp4477 & 1))
      {
        tmp4474 = -pow(-tmp4472, tmp4475)*pow(tmp4472, tmp4476);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4472, tmp4473);
      }
    }
  }
  else
  {
    tmp4474 = pow(tmp4472, tmp4473);
  }
  if(isnan(tmp4474) || isinf(tmp4474))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4472, tmp4473);
  }tmp4479 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4474,"(r_init[53] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4479 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[53] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4479);
    }
  }
  (data->simulationInfo->realParameter[557] /* omega_c[53] PARAM */) = sqrt(tmp4479);
  TRACE_POP
}

/*
equation index: 13897
type: SIMPLE_ASSIGN
r_init[52] = r_min + 52.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13897};
  (data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (52.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13898
type: SIMPLE_ASSIGN
omega_c[52] = sqrt(G * Md / (r_init[52] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13898(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13898};
  modelica_real tmp4480;
  modelica_real tmp4481;
  modelica_real tmp4482;
  modelica_real tmp4483;
  modelica_real tmp4484;
  modelica_real tmp4485;
  modelica_real tmp4486;
  modelica_real tmp4487;
  modelica_real tmp4488;
  modelica_real tmp4489;
  tmp4480 = (data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */);
  tmp4481 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4482 = (tmp4480 * tmp4480) + (tmp4481 * tmp4481);
  tmp4483 = 1.5;
  if(tmp4482 < 0.0 && tmp4483 != 0.0)
  {
    tmp4485 = modf(tmp4483, &tmp4486);
    
    if(tmp4485 > 0.5)
    {
      tmp4485 -= 1.0;
      tmp4486 += 1.0;
    }
    else if(tmp4485 < -0.5)
    {
      tmp4485 += 1.0;
      tmp4486 -= 1.0;
    }
    
    if(fabs(tmp4485) < 1e-10)
      tmp4484 = pow(tmp4482, tmp4486);
    else
    {
      tmp4488 = modf(1.0/tmp4483, &tmp4487);
      if(tmp4488 > 0.5)
      {
        tmp4488 -= 1.0;
        tmp4487 += 1.0;
      }
      else if(tmp4488 < -0.5)
      {
        tmp4488 += 1.0;
        tmp4487 -= 1.0;
      }
      if(fabs(tmp4488) < 1e-10 && ((unsigned long)tmp4487 & 1))
      {
        tmp4484 = -pow(-tmp4482, tmp4485)*pow(tmp4482, tmp4486);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4482, tmp4483);
      }
    }
  }
  else
  {
    tmp4484 = pow(tmp4482, tmp4483);
  }
  if(isnan(tmp4484) || isinf(tmp4484))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4482, tmp4483);
  }tmp4489 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4484,"(r_init[52] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4489 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[52] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4489);
    }
  }
  (data->simulationInfo->realParameter[556] /* omega_c[52] PARAM */) = sqrt(tmp4489);
  TRACE_POP
}

/*
equation index: 13899
type: SIMPLE_ASSIGN
r_init[51] = r_min + 51.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13899};
  (data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (51.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13900
type: SIMPLE_ASSIGN
omega_c[51] = sqrt(G * Md / (r_init[51] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13900};
  modelica_real tmp4490;
  modelica_real tmp4491;
  modelica_real tmp4492;
  modelica_real tmp4493;
  modelica_real tmp4494;
  modelica_real tmp4495;
  modelica_real tmp4496;
  modelica_real tmp4497;
  modelica_real tmp4498;
  modelica_real tmp4499;
  tmp4490 = (data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */);
  tmp4491 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4492 = (tmp4490 * tmp4490) + (tmp4491 * tmp4491);
  tmp4493 = 1.5;
  if(tmp4492 < 0.0 && tmp4493 != 0.0)
  {
    tmp4495 = modf(tmp4493, &tmp4496);
    
    if(tmp4495 > 0.5)
    {
      tmp4495 -= 1.0;
      tmp4496 += 1.0;
    }
    else if(tmp4495 < -0.5)
    {
      tmp4495 += 1.0;
      tmp4496 -= 1.0;
    }
    
    if(fabs(tmp4495) < 1e-10)
      tmp4494 = pow(tmp4492, tmp4496);
    else
    {
      tmp4498 = modf(1.0/tmp4493, &tmp4497);
      if(tmp4498 > 0.5)
      {
        tmp4498 -= 1.0;
        tmp4497 += 1.0;
      }
      else if(tmp4498 < -0.5)
      {
        tmp4498 += 1.0;
        tmp4497 -= 1.0;
      }
      if(fabs(tmp4498) < 1e-10 && ((unsigned long)tmp4497 & 1))
      {
        tmp4494 = -pow(-tmp4492, tmp4495)*pow(tmp4492, tmp4496);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4492, tmp4493);
      }
    }
  }
  else
  {
    tmp4494 = pow(tmp4492, tmp4493);
  }
  if(isnan(tmp4494) || isinf(tmp4494))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4492, tmp4493);
  }tmp4499 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4494,"(r_init[51] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4499 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[51] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4499);
    }
  }
  (data->simulationInfo->realParameter[555] /* omega_c[51] PARAM */) = sqrt(tmp4499);
  TRACE_POP
}

/*
equation index: 13901
type: SIMPLE_ASSIGN
r_init[50] = r_min + 50.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13901};
  (data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (50.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13902
type: SIMPLE_ASSIGN
omega_c[50] = sqrt(G * Md / (r_init[50] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13902(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13902};
  modelica_real tmp4500;
  modelica_real tmp4501;
  modelica_real tmp4502;
  modelica_real tmp4503;
  modelica_real tmp4504;
  modelica_real tmp4505;
  modelica_real tmp4506;
  modelica_real tmp4507;
  modelica_real tmp4508;
  modelica_real tmp4509;
  tmp4500 = (data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */);
  tmp4501 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4502 = (tmp4500 * tmp4500) + (tmp4501 * tmp4501);
  tmp4503 = 1.5;
  if(tmp4502 < 0.0 && tmp4503 != 0.0)
  {
    tmp4505 = modf(tmp4503, &tmp4506);
    
    if(tmp4505 > 0.5)
    {
      tmp4505 -= 1.0;
      tmp4506 += 1.0;
    }
    else if(tmp4505 < -0.5)
    {
      tmp4505 += 1.0;
      tmp4506 -= 1.0;
    }
    
    if(fabs(tmp4505) < 1e-10)
      tmp4504 = pow(tmp4502, tmp4506);
    else
    {
      tmp4508 = modf(1.0/tmp4503, &tmp4507);
      if(tmp4508 > 0.5)
      {
        tmp4508 -= 1.0;
        tmp4507 += 1.0;
      }
      else if(tmp4508 < -0.5)
      {
        tmp4508 += 1.0;
        tmp4507 -= 1.0;
      }
      if(fabs(tmp4508) < 1e-10 && ((unsigned long)tmp4507 & 1))
      {
        tmp4504 = -pow(-tmp4502, tmp4505)*pow(tmp4502, tmp4506);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4502, tmp4503);
      }
    }
  }
  else
  {
    tmp4504 = pow(tmp4502, tmp4503);
  }
  if(isnan(tmp4504) || isinf(tmp4504))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4502, tmp4503);
  }tmp4509 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4504,"(r_init[50] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4509 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[50] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4509);
    }
  }
  (data->simulationInfo->realParameter[554] /* omega_c[50] PARAM */) = sqrt(tmp4509);
  TRACE_POP
}

/*
equation index: 13903
type: SIMPLE_ASSIGN
r_init[49] = r_min + 49.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13903};
  (data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (49.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13904
type: SIMPLE_ASSIGN
omega_c[49] = sqrt(G * Md / (r_init[49] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13904(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13904};
  modelica_real tmp4510;
  modelica_real tmp4511;
  modelica_real tmp4512;
  modelica_real tmp4513;
  modelica_real tmp4514;
  modelica_real tmp4515;
  modelica_real tmp4516;
  modelica_real tmp4517;
  modelica_real tmp4518;
  modelica_real tmp4519;
  tmp4510 = (data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */);
  tmp4511 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4512 = (tmp4510 * tmp4510) + (tmp4511 * tmp4511);
  tmp4513 = 1.5;
  if(tmp4512 < 0.0 && tmp4513 != 0.0)
  {
    tmp4515 = modf(tmp4513, &tmp4516);
    
    if(tmp4515 > 0.5)
    {
      tmp4515 -= 1.0;
      tmp4516 += 1.0;
    }
    else if(tmp4515 < -0.5)
    {
      tmp4515 += 1.0;
      tmp4516 -= 1.0;
    }
    
    if(fabs(tmp4515) < 1e-10)
      tmp4514 = pow(tmp4512, tmp4516);
    else
    {
      tmp4518 = modf(1.0/tmp4513, &tmp4517);
      if(tmp4518 > 0.5)
      {
        tmp4518 -= 1.0;
        tmp4517 += 1.0;
      }
      else if(tmp4518 < -0.5)
      {
        tmp4518 += 1.0;
        tmp4517 -= 1.0;
      }
      if(fabs(tmp4518) < 1e-10 && ((unsigned long)tmp4517 & 1))
      {
        tmp4514 = -pow(-tmp4512, tmp4515)*pow(tmp4512, tmp4516);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4512, tmp4513);
      }
    }
  }
  else
  {
    tmp4514 = pow(tmp4512, tmp4513);
  }
  if(isnan(tmp4514) || isinf(tmp4514))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4512, tmp4513);
  }tmp4519 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4514,"(r_init[49] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4519 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[49] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4519);
    }
  }
  (data->simulationInfo->realParameter[553] /* omega_c[49] PARAM */) = sqrt(tmp4519);
  TRACE_POP
}

/*
equation index: 13905
type: SIMPLE_ASSIGN
r_init[48] = r_min + 48.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13905};
  (data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (48.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13906
type: SIMPLE_ASSIGN
omega_c[48] = sqrt(G * Md / (r_init[48] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13906(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13906};
  modelica_real tmp4520;
  modelica_real tmp4521;
  modelica_real tmp4522;
  modelica_real tmp4523;
  modelica_real tmp4524;
  modelica_real tmp4525;
  modelica_real tmp4526;
  modelica_real tmp4527;
  modelica_real tmp4528;
  modelica_real tmp4529;
  tmp4520 = (data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */);
  tmp4521 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4522 = (tmp4520 * tmp4520) + (tmp4521 * tmp4521);
  tmp4523 = 1.5;
  if(tmp4522 < 0.0 && tmp4523 != 0.0)
  {
    tmp4525 = modf(tmp4523, &tmp4526);
    
    if(tmp4525 > 0.5)
    {
      tmp4525 -= 1.0;
      tmp4526 += 1.0;
    }
    else if(tmp4525 < -0.5)
    {
      tmp4525 += 1.0;
      tmp4526 -= 1.0;
    }
    
    if(fabs(tmp4525) < 1e-10)
      tmp4524 = pow(tmp4522, tmp4526);
    else
    {
      tmp4528 = modf(1.0/tmp4523, &tmp4527);
      if(tmp4528 > 0.5)
      {
        tmp4528 -= 1.0;
        tmp4527 += 1.0;
      }
      else if(tmp4528 < -0.5)
      {
        tmp4528 += 1.0;
        tmp4527 -= 1.0;
      }
      if(fabs(tmp4528) < 1e-10 && ((unsigned long)tmp4527 & 1))
      {
        tmp4524 = -pow(-tmp4522, tmp4525)*pow(tmp4522, tmp4526);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4522, tmp4523);
      }
    }
  }
  else
  {
    tmp4524 = pow(tmp4522, tmp4523);
  }
  if(isnan(tmp4524) || isinf(tmp4524))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4522, tmp4523);
  }tmp4529 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4524,"(r_init[48] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4529 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[48] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4529);
    }
  }
  (data->simulationInfo->realParameter[552] /* omega_c[48] PARAM */) = sqrt(tmp4529);
  TRACE_POP
}

/*
equation index: 13907
type: SIMPLE_ASSIGN
r_init[47] = r_min + 47.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13907};
  (data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (47.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13908
type: SIMPLE_ASSIGN
omega_c[47] = sqrt(G * Md / (r_init[47] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13908};
  modelica_real tmp4530;
  modelica_real tmp4531;
  modelica_real tmp4532;
  modelica_real tmp4533;
  modelica_real tmp4534;
  modelica_real tmp4535;
  modelica_real tmp4536;
  modelica_real tmp4537;
  modelica_real tmp4538;
  modelica_real tmp4539;
  tmp4530 = (data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */);
  tmp4531 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4532 = (tmp4530 * tmp4530) + (tmp4531 * tmp4531);
  tmp4533 = 1.5;
  if(tmp4532 < 0.0 && tmp4533 != 0.0)
  {
    tmp4535 = modf(tmp4533, &tmp4536);
    
    if(tmp4535 > 0.5)
    {
      tmp4535 -= 1.0;
      tmp4536 += 1.0;
    }
    else if(tmp4535 < -0.5)
    {
      tmp4535 += 1.0;
      tmp4536 -= 1.0;
    }
    
    if(fabs(tmp4535) < 1e-10)
      tmp4534 = pow(tmp4532, tmp4536);
    else
    {
      tmp4538 = modf(1.0/tmp4533, &tmp4537);
      if(tmp4538 > 0.5)
      {
        tmp4538 -= 1.0;
        tmp4537 += 1.0;
      }
      else if(tmp4538 < -0.5)
      {
        tmp4538 += 1.0;
        tmp4537 -= 1.0;
      }
      if(fabs(tmp4538) < 1e-10 && ((unsigned long)tmp4537 & 1))
      {
        tmp4534 = -pow(-tmp4532, tmp4535)*pow(tmp4532, tmp4536);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4532, tmp4533);
      }
    }
  }
  else
  {
    tmp4534 = pow(tmp4532, tmp4533);
  }
  if(isnan(tmp4534) || isinf(tmp4534))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4532, tmp4533);
  }tmp4539 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4534,"(r_init[47] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4539 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[47] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4539);
    }
  }
  (data->simulationInfo->realParameter[551] /* omega_c[47] PARAM */) = sqrt(tmp4539);
  TRACE_POP
}

/*
equation index: 13909
type: SIMPLE_ASSIGN
r_init[46] = r_min + 46.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13909};
  (data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (46.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13910
type: SIMPLE_ASSIGN
omega_c[46] = sqrt(G * Md / (r_init[46] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13910(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13910};
  modelica_real tmp4540;
  modelica_real tmp4541;
  modelica_real tmp4542;
  modelica_real tmp4543;
  modelica_real tmp4544;
  modelica_real tmp4545;
  modelica_real tmp4546;
  modelica_real tmp4547;
  modelica_real tmp4548;
  modelica_real tmp4549;
  tmp4540 = (data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */);
  tmp4541 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4542 = (tmp4540 * tmp4540) + (tmp4541 * tmp4541);
  tmp4543 = 1.5;
  if(tmp4542 < 0.0 && tmp4543 != 0.0)
  {
    tmp4545 = modf(tmp4543, &tmp4546);
    
    if(tmp4545 > 0.5)
    {
      tmp4545 -= 1.0;
      tmp4546 += 1.0;
    }
    else if(tmp4545 < -0.5)
    {
      tmp4545 += 1.0;
      tmp4546 -= 1.0;
    }
    
    if(fabs(tmp4545) < 1e-10)
      tmp4544 = pow(tmp4542, tmp4546);
    else
    {
      tmp4548 = modf(1.0/tmp4543, &tmp4547);
      if(tmp4548 > 0.5)
      {
        tmp4548 -= 1.0;
        tmp4547 += 1.0;
      }
      else if(tmp4548 < -0.5)
      {
        tmp4548 += 1.0;
        tmp4547 -= 1.0;
      }
      if(fabs(tmp4548) < 1e-10 && ((unsigned long)tmp4547 & 1))
      {
        tmp4544 = -pow(-tmp4542, tmp4545)*pow(tmp4542, tmp4546);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4542, tmp4543);
      }
    }
  }
  else
  {
    tmp4544 = pow(tmp4542, tmp4543);
  }
  if(isnan(tmp4544) || isinf(tmp4544))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4542, tmp4543);
  }tmp4549 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4544,"(r_init[46] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4549 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[46] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4549);
    }
  }
  (data->simulationInfo->realParameter[550] /* omega_c[46] PARAM */) = sqrt(tmp4549);
  TRACE_POP
}

/*
equation index: 13911
type: SIMPLE_ASSIGN
r_init[45] = r_min + 45.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13911};
  (data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (45.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13912
type: SIMPLE_ASSIGN
omega_c[45] = sqrt(G * Md / (r_init[45] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13912(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13912};
  modelica_real tmp4550;
  modelica_real tmp4551;
  modelica_real tmp4552;
  modelica_real tmp4553;
  modelica_real tmp4554;
  modelica_real tmp4555;
  modelica_real tmp4556;
  modelica_real tmp4557;
  modelica_real tmp4558;
  modelica_real tmp4559;
  tmp4550 = (data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */);
  tmp4551 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4552 = (tmp4550 * tmp4550) + (tmp4551 * tmp4551);
  tmp4553 = 1.5;
  if(tmp4552 < 0.0 && tmp4553 != 0.0)
  {
    tmp4555 = modf(tmp4553, &tmp4556);
    
    if(tmp4555 > 0.5)
    {
      tmp4555 -= 1.0;
      tmp4556 += 1.0;
    }
    else if(tmp4555 < -0.5)
    {
      tmp4555 += 1.0;
      tmp4556 -= 1.0;
    }
    
    if(fabs(tmp4555) < 1e-10)
      tmp4554 = pow(tmp4552, tmp4556);
    else
    {
      tmp4558 = modf(1.0/tmp4553, &tmp4557);
      if(tmp4558 > 0.5)
      {
        tmp4558 -= 1.0;
        tmp4557 += 1.0;
      }
      else if(tmp4558 < -0.5)
      {
        tmp4558 += 1.0;
        tmp4557 -= 1.0;
      }
      if(fabs(tmp4558) < 1e-10 && ((unsigned long)tmp4557 & 1))
      {
        tmp4554 = -pow(-tmp4552, tmp4555)*pow(tmp4552, tmp4556);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4552, tmp4553);
      }
    }
  }
  else
  {
    tmp4554 = pow(tmp4552, tmp4553);
  }
  if(isnan(tmp4554) || isinf(tmp4554))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4552, tmp4553);
  }tmp4559 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4554,"(r_init[45] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4559 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[45] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4559);
    }
  }
  (data->simulationInfo->realParameter[549] /* omega_c[45] PARAM */) = sqrt(tmp4559);
  TRACE_POP
}

/*
equation index: 13913
type: SIMPLE_ASSIGN
r_init[44] = r_min + 44.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13913};
  (data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (44.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13914
type: SIMPLE_ASSIGN
omega_c[44] = sqrt(G * Md / (r_init[44] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13914(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13914};
  modelica_real tmp4560;
  modelica_real tmp4561;
  modelica_real tmp4562;
  modelica_real tmp4563;
  modelica_real tmp4564;
  modelica_real tmp4565;
  modelica_real tmp4566;
  modelica_real tmp4567;
  modelica_real tmp4568;
  modelica_real tmp4569;
  tmp4560 = (data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */);
  tmp4561 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4562 = (tmp4560 * tmp4560) + (tmp4561 * tmp4561);
  tmp4563 = 1.5;
  if(tmp4562 < 0.0 && tmp4563 != 0.0)
  {
    tmp4565 = modf(tmp4563, &tmp4566);
    
    if(tmp4565 > 0.5)
    {
      tmp4565 -= 1.0;
      tmp4566 += 1.0;
    }
    else if(tmp4565 < -0.5)
    {
      tmp4565 += 1.0;
      tmp4566 -= 1.0;
    }
    
    if(fabs(tmp4565) < 1e-10)
      tmp4564 = pow(tmp4562, tmp4566);
    else
    {
      tmp4568 = modf(1.0/tmp4563, &tmp4567);
      if(tmp4568 > 0.5)
      {
        tmp4568 -= 1.0;
        tmp4567 += 1.0;
      }
      else if(tmp4568 < -0.5)
      {
        tmp4568 += 1.0;
        tmp4567 -= 1.0;
      }
      if(fabs(tmp4568) < 1e-10 && ((unsigned long)tmp4567 & 1))
      {
        tmp4564 = -pow(-tmp4562, tmp4565)*pow(tmp4562, tmp4566);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4562, tmp4563);
      }
    }
  }
  else
  {
    tmp4564 = pow(tmp4562, tmp4563);
  }
  if(isnan(tmp4564) || isinf(tmp4564))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4562, tmp4563);
  }tmp4569 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4564,"(r_init[44] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4569 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[44] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4569);
    }
  }
  (data->simulationInfo->realParameter[548] /* omega_c[44] PARAM */) = sqrt(tmp4569);
  TRACE_POP
}

/*
equation index: 13915
type: SIMPLE_ASSIGN
r_init[43] = r_min + 43.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13915};
  (data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (43.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13916
type: SIMPLE_ASSIGN
omega_c[43] = sqrt(G * Md / (r_init[43] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13916};
  modelica_real tmp4570;
  modelica_real tmp4571;
  modelica_real tmp4572;
  modelica_real tmp4573;
  modelica_real tmp4574;
  modelica_real tmp4575;
  modelica_real tmp4576;
  modelica_real tmp4577;
  modelica_real tmp4578;
  modelica_real tmp4579;
  tmp4570 = (data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */);
  tmp4571 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4572 = (tmp4570 * tmp4570) + (tmp4571 * tmp4571);
  tmp4573 = 1.5;
  if(tmp4572 < 0.0 && tmp4573 != 0.0)
  {
    tmp4575 = modf(tmp4573, &tmp4576);
    
    if(tmp4575 > 0.5)
    {
      tmp4575 -= 1.0;
      tmp4576 += 1.0;
    }
    else if(tmp4575 < -0.5)
    {
      tmp4575 += 1.0;
      tmp4576 -= 1.0;
    }
    
    if(fabs(tmp4575) < 1e-10)
      tmp4574 = pow(tmp4572, tmp4576);
    else
    {
      tmp4578 = modf(1.0/tmp4573, &tmp4577);
      if(tmp4578 > 0.5)
      {
        tmp4578 -= 1.0;
        tmp4577 += 1.0;
      }
      else if(tmp4578 < -0.5)
      {
        tmp4578 += 1.0;
        tmp4577 -= 1.0;
      }
      if(fabs(tmp4578) < 1e-10 && ((unsigned long)tmp4577 & 1))
      {
        tmp4574 = -pow(-tmp4572, tmp4575)*pow(tmp4572, tmp4576);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4572, tmp4573);
      }
    }
  }
  else
  {
    tmp4574 = pow(tmp4572, tmp4573);
  }
  if(isnan(tmp4574) || isinf(tmp4574))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4572, tmp4573);
  }tmp4579 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4574,"(r_init[43] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4579 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[43] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4579);
    }
  }
  (data->simulationInfo->realParameter[547] /* omega_c[43] PARAM */) = sqrt(tmp4579);
  TRACE_POP
}

/*
equation index: 13917
type: SIMPLE_ASSIGN
r_init[42] = r_min + 42.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13917};
  (data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (42.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13918
type: SIMPLE_ASSIGN
omega_c[42] = sqrt(G * Md / (r_init[42] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13918(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13918};
  modelica_real tmp4580;
  modelica_real tmp4581;
  modelica_real tmp4582;
  modelica_real tmp4583;
  modelica_real tmp4584;
  modelica_real tmp4585;
  modelica_real tmp4586;
  modelica_real tmp4587;
  modelica_real tmp4588;
  modelica_real tmp4589;
  tmp4580 = (data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */);
  tmp4581 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4582 = (tmp4580 * tmp4580) + (tmp4581 * tmp4581);
  tmp4583 = 1.5;
  if(tmp4582 < 0.0 && tmp4583 != 0.0)
  {
    tmp4585 = modf(tmp4583, &tmp4586);
    
    if(tmp4585 > 0.5)
    {
      tmp4585 -= 1.0;
      tmp4586 += 1.0;
    }
    else if(tmp4585 < -0.5)
    {
      tmp4585 += 1.0;
      tmp4586 -= 1.0;
    }
    
    if(fabs(tmp4585) < 1e-10)
      tmp4584 = pow(tmp4582, tmp4586);
    else
    {
      tmp4588 = modf(1.0/tmp4583, &tmp4587);
      if(tmp4588 > 0.5)
      {
        tmp4588 -= 1.0;
        tmp4587 += 1.0;
      }
      else if(tmp4588 < -0.5)
      {
        tmp4588 += 1.0;
        tmp4587 -= 1.0;
      }
      if(fabs(tmp4588) < 1e-10 && ((unsigned long)tmp4587 & 1))
      {
        tmp4584 = -pow(-tmp4582, tmp4585)*pow(tmp4582, tmp4586);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4582, tmp4583);
      }
    }
  }
  else
  {
    tmp4584 = pow(tmp4582, tmp4583);
  }
  if(isnan(tmp4584) || isinf(tmp4584))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4582, tmp4583);
  }tmp4589 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4584,"(r_init[42] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4589 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[42] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4589);
    }
  }
  (data->simulationInfo->realParameter[546] /* omega_c[42] PARAM */) = sqrt(tmp4589);
  TRACE_POP
}

/*
equation index: 13919
type: SIMPLE_ASSIGN
r_init[41] = r_min + 41.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13919};
  (data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (41.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13920
type: SIMPLE_ASSIGN
omega_c[41] = sqrt(G * Md / (r_init[41] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13920(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13920};
  modelica_real tmp4590;
  modelica_real tmp4591;
  modelica_real tmp4592;
  modelica_real tmp4593;
  modelica_real tmp4594;
  modelica_real tmp4595;
  modelica_real tmp4596;
  modelica_real tmp4597;
  modelica_real tmp4598;
  modelica_real tmp4599;
  tmp4590 = (data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */);
  tmp4591 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4592 = (tmp4590 * tmp4590) + (tmp4591 * tmp4591);
  tmp4593 = 1.5;
  if(tmp4592 < 0.0 && tmp4593 != 0.0)
  {
    tmp4595 = modf(tmp4593, &tmp4596);
    
    if(tmp4595 > 0.5)
    {
      tmp4595 -= 1.0;
      tmp4596 += 1.0;
    }
    else if(tmp4595 < -0.5)
    {
      tmp4595 += 1.0;
      tmp4596 -= 1.0;
    }
    
    if(fabs(tmp4595) < 1e-10)
      tmp4594 = pow(tmp4592, tmp4596);
    else
    {
      tmp4598 = modf(1.0/tmp4593, &tmp4597);
      if(tmp4598 > 0.5)
      {
        tmp4598 -= 1.0;
        tmp4597 += 1.0;
      }
      else if(tmp4598 < -0.5)
      {
        tmp4598 += 1.0;
        tmp4597 -= 1.0;
      }
      if(fabs(tmp4598) < 1e-10 && ((unsigned long)tmp4597 & 1))
      {
        tmp4594 = -pow(-tmp4592, tmp4595)*pow(tmp4592, tmp4596);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4592, tmp4593);
      }
    }
  }
  else
  {
    tmp4594 = pow(tmp4592, tmp4593);
  }
  if(isnan(tmp4594) || isinf(tmp4594))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4592, tmp4593);
  }tmp4599 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4594,"(r_init[41] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4599 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[41] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4599);
    }
  }
  (data->simulationInfo->realParameter[545] /* omega_c[41] PARAM */) = sqrt(tmp4599);
  TRACE_POP
}

/*
equation index: 13921
type: SIMPLE_ASSIGN
r_init[40] = r_min + 40.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13921};
  (data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (40.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13922
type: SIMPLE_ASSIGN
omega_c[40] = sqrt(G * Md / (r_init[40] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13922(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13922};
  modelica_real tmp4600;
  modelica_real tmp4601;
  modelica_real tmp4602;
  modelica_real tmp4603;
  modelica_real tmp4604;
  modelica_real tmp4605;
  modelica_real tmp4606;
  modelica_real tmp4607;
  modelica_real tmp4608;
  modelica_real tmp4609;
  tmp4600 = (data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */);
  tmp4601 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4602 = (tmp4600 * tmp4600) + (tmp4601 * tmp4601);
  tmp4603 = 1.5;
  if(tmp4602 < 0.0 && tmp4603 != 0.0)
  {
    tmp4605 = modf(tmp4603, &tmp4606);
    
    if(tmp4605 > 0.5)
    {
      tmp4605 -= 1.0;
      tmp4606 += 1.0;
    }
    else if(tmp4605 < -0.5)
    {
      tmp4605 += 1.0;
      tmp4606 -= 1.0;
    }
    
    if(fabs(tmp4605) < 1e-10)
      tmp4604 = pow(tmp4602, tmp4606);
    else
    {
      tmp4608 = modf(1.0/tmp4603, &tmp4607);
      if(tmp4608 > 0.5)
      {
        tmp4608 -= 1.0;
        tmp4607 += 1.0;
      }
      else if(tmp4608 < -0.5)
      {
        tmp4608 += 1.0;
        tmp4607 -= 1.0;
      }
      if(fabs(tmp4608) < 1e-10 && ((unsigned long)tmp4607 & 1))
      {
        tmp4604 = -pow(-tmp4602, tmp4605)*pow(tmp4602, tmp4606);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4602, tmp4603);
      }
    }
  }
  else
  {
    tmp4604 = pow(tmp4602, tmp4603);
  }
  if(isnan(tmp4604) || isinf(tmp4604))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4602, tmp4603);
  }tmp4609 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4604,"(r_init[40] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4609 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[40] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4609);
    }
  }
  (data->simulationInfo->realParameter[544] /* omega_c[40] PARAM */) = sqrt(tmp4609);
  TRACE_POP
}

/*
equation index: 13923
type: SIMPLE_ASSIGN
r_init[39] = r_min + 39.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13923};
  (data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (39.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13924
type: SIMPLE_ASSIGN
omega_c[39] = sqrt(G * Md / (r_init[39] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13924};
  modelica_real tmp4610;
  modelica_real tmp4611;
  modelica_real tmp4612;
  modelica_real tmp4613;
  modelica_real tmp4614;
  modelica_real tmp4615;
  modelica_real tmp4616;
  modelica_real tmp4617;
  modelica_real tmp4618;
  modelica_real tmp4619;
  tmp4610 = (data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */);
  tmp4611 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4612 = (tmp4610 * tmp4610) + (tmp4611 * tmp4611);
  tmp4613 = 1.5;
  if(tmp4612 < 0.0 && tmp4613 != 0.0)
  {
    tmp4615 = modf(tmp4613, &tmp4616);
    
    if(tmp4615 > 0.5)
    {
      tmp4615 -= 1.0;
      tmp4616 += 1.0;
    }
    else if(tmp4615 < -0.5)
    {
      tmp4615 += 1.0;
      tmp4616 -= 1.0;
    }
    
    if(fabs(tmp4615) < 1e-10)
      tmp4614 = pow(tmp4612, tmp4616);
    else
    {
      tmp4618 = modf(1.0/tmp4613, &tmp4617);
      if(tmp4618 > 0.5)
      {
        tmp4618 -= 1.0;
        tmp4617 += 1.0;
      }
      else if(tmp4618 < -0.5)
      {
        tmp4618 += 1.0;
        tmp4617 -= 1.0;
      }
      if(fabs(tmp4618) < 1e-10 && ((unsigned long)tmp4617 & 1))
      {
        tmp4614 = -pow(-tmp4612, tmp4615)*pow(tmp4612, tmp4616);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4612, tmp4613);
      }
    }
  }
  else
  {
    tmp4614 = pow(tmp4612, tmp4613);
  }
  if(isnan(tmp4614) || isinf(tmp4614))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4612, tmp4613);
  }tmp4619 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4614,"(r_init[39] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4619 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[39] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4619);
    }
  }
  (data->simulationInfo->realParameter[543] /* omega_c[39] PARAM */) = sqrt(tmp4619);
  TRACE_POP
}

/*
equation index: 13925
type: SIMPLE_ASSIGN
r_init[38] = r_min + 38.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13925};
  (data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (38.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13926
type: SIMPLE_ASSIGN
omega_c[38] = sqrt(G * Md / (r_init[38] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13926(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13926};
  modelica_real tmp4620;
  modelica_real tmp4621;
  modelica_real tmp4622;
  modelica_real tmp4623;
  modelica_real tmp4624;
  modelica_real tmp4625;
  modelica_real tmp4626;
  modelica_real tmp4627;
  modelica_real tmp4628;
  modelica_real tmp4629;
  tmp4620 = (data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */);
  tmp4621 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4622 = (tmp4620 * tmp4620) + (tmp4621 * tmp4621);
  tmp4623 = 1.5;
  if(tmp4622 < 0.0 && tmp4623 != 0.0)
  {
    tmp4625 = modf(tmp4623, &tmp4626);
    
    if(tmp4625 > 0.5)
    {
      tmp4625 -= 1.0;
      tmp4626 += 1.0;
    }
    else if(tmp4625 < -0.5)
    {
      tmp4625 += 1.0;
      tmp4626 -= 1.0;
    }
    
    if(fabs(tmp4625) < 1e-10)
      tmp4624 = pow(tmp4622, tmp4626);
    else
    {
      tmp4628 = modf(1.0/tmp4623, &tmp4627);
      if(tmp4628 > 0.5)
      {
        tmp4628 -= 1.0;
        tmp4627 += 1.0;
      }
      else if(tmp4628 < -0.5)
      {
        tmp4628 += 1.0;
        tmp4627 -= 1.0;
      }
      if(fabs(tmp4628) < 1e-10 && ((unsigned long)tmp4627 & 1))
      {
        tmp4624 = -pow(-tmp4622, tmp4625)*pow(tmp4622, tmp4626);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4622, tmp4623);
      }
    }
  }
  else
  {
    tmp4624 = pow(tmp4622, tmp4623);
  }
  if(isnan(tmp4624) || isinf(tmp4624))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4622, tmp4623);
  }tmp4629 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4624,"(r_init[38] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4629 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[38] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4629);
    }
  }
  (data->simulationInfo->realParameter[542] /* omega_c[38] PARAM */) = sqrt(tmp4629);
  TRACE_POP
}

/*
equation index: 13927
type: SIMPLE_ASSIGN
r_init[37] = r_min + 37.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13927};
  (data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (37.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13928
type: SIMPLE_ASSIGN
omega_c[37] = sqrt(G * Md / (r_init[37] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13928(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13928};
  modelica_real tmp4630;
  modelica_real tmp4631;
  modelica_real tmp4632;
  modelica_real tmp4633;
  modelica_real tmp4634;
  modelica_real tmp4635;
  modelica_real tmp4636;
  modelica_real tmp4637;
  modelica_real tmp4638;
  modelica_real tmp4639;
  tmp4630 = (data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */);
  tmp4631 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4632 = (tmp4630 * tmp4630) + (tmp4631 * tmp4631);
  tmp4633 = 1.5;
  if(tmp4632 < 0.0 && tmp4633 != 0.0)
  {
    tmp4635 = modf(tmp4633, &tmp4636);
    
    if(tmp4635 > 0.5)
    {
      tmp4635 -= 1.0;
      tmp4636 += 1.0;
    }
    else if(tmp4635 < -0.5)
    {
      tmp4635 += 1.0;
      tmp4636 -= 1.0;
    }
    
    if(fabs(tmp4635) < 1e-10)
      tmp4634 = pow(tmp4632, tmp4636);
    else
    {
      tmp4638 = modf(1.0/tmp4633, &tmp4637);
      if(tmp4638 > 0.5)
      {
        tmp4638 -= 1.0;
        tmp4637 += 1.0;
      }
      else if(tmp4638 < -0.5)
      {
        tmp4638 += 1.0;
        tmp4637 -= 1.0;
      }
      if(fabs(tmp4638) < 1e-10 && ((unsigned long)tmp4637 & 1))
      {
        tmp4634 = -pow(-tmp4632, tmp4635)*pow(tmp4632, tmp4636);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4632, tmp4633);
      }
    }
  }
  else
  {
    tmp4634 = pow(tmp4632, tmp4633);
  }
  if(isnan(tmp4634) || isinf(tmp4634))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4632, tmp4633);
  }tmp4639 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4634,"(r_init[37] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4639 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[37] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4639);
    }
  }
  (data->simulationInfo->realParameter[541] /* omega_c[37] PARAM */) = sqrt(tmp4639);
  TRACE_POP
}

/*
equation index: 13929
type: SIMPLE_ASSIGN
r_init[36] = r_min + 36.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13929};
  (data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (36.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13930
type: SIMPLE_ASSIGN
omega_c[36] = sqrt(G * Md / (r_init[36] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13930(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13930};
  modelica_real tmp4640;
  modelica_real tmp4641;
  modelica_real tmp4642;
  modelica_real tmp4643;
  modelica_real tmp4644;
  modelica_real tmp4645;
  modelica_real tmp4646;
  modelica_real tmp4647;
  modelica_real tmp4648;
  modelica_real tmp4649;
  tmp4640 = (data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */);
  tmp4641 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4642 = (tmp4640 * tmp4640) + (tmp4641 * tmp4641);
  tmp4643 = 1.5;
  if(tmp4642 < 0.0 && tmp4643 != 0.0)
  {
    tmp4645 = modf(tmp4643, &tmp4646);
    
    if(tmp4645 > 0.5)
    {
      tmp4645 -= 1.0;
      tmp4646 += 1.0;
    }
    else if(tmp4645 < -0.5)
    {
      tmp4645 += 1.0;
      tmp4646 -= 1.0;
    }
    
    if(fabs(tmp4645) < 1e-10)
      tmp4644 = pow(tmp4642, tmp4646);
    else
    {
      tmp4648 = modf(1.0/tmp4643, &tmp4647);
      if(tmp4648 > 0.5)
      {
        tmp4648 -= 1.0;
        tmp4647 += 1.0;
      }
      else if(tmp4648 < -0.5)
      {
        tmp4648 += 1.0;
        tmp4647 -= 1.0;
      }
      if(fabs(tmp4648) < 1e-10 && ((unsigned long)tmp4647 & 1))
      {
        tmp4644 = -pow(-tmp4642, tmp4645)*pow(tmp4642, tmp4646);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4642, tmp4643);
      }
    }
  }
  else
  {
    tmp4644 = pow(tmp4642, tmp4643);
  }
  if(isnan(tmp4644) || isinf(tmp4644))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4642, tmp4643);
  }tmp4649 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4644,"(r_init[36] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4649 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[36] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4649);
    }
  }
  (data->simulationInfo->realParameter[540] /* omega_c[36] PARAM */) = sqrt(tmp4649);
  TRACE_POP
}

/*
equation index: 13931
type: SIMPLE_ASSIGN
r_init[35] = r_min + 35.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13931};
  (data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (35.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13932
type: SIMPLE_ASSIGN
omega_c[35] = sqrt(G * Md / (r_init[35] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13932};
  modelica_real tmp4650;
  modelica_real tmp4651;
  modelica_real tmp4652;
  modelica_real tmp4653;
  modelica_real tmp4654;
  modelica_real tmp4655;
  modelica_real tmp4656;
  modelica_real tmp4657;
  modelica_real tmp4658;
  modelica_real tmp4659;
  tmp4650 = (data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */);
  tmp4651 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4652 = (tmp4650 * tmp4650) + (tmp4651 * tmp4651);
  tmp4653 = 1.5;
  if(tmp4652 < 0.0 && tmp4653 != 0.0)
  {
    tmp4655 = modf(tmp4653, &tmp4656);
    
    if(tmp4655 > 0.5)
    {
      tmp4655 -= 1.0;
      tmp4656 += 1.0;
    }
    else if(tmp4655 < -0.5)
    {
      tmp4655 += 1.0;
      tmp4656 -= 1.0;
    }
    
    if(fabs(tmp4655) < 1e-10)
      tmp4654 = pow(tmp4652, tmp4656);
    else
    {
      tmp4658 = modf(1.0/tmp4653, &tmp4657);
      if(tmp4658 > 0.5)
      {
        tmp4658 -= 1.0;
        tmp4657 += 1.0;
      }
      else if(tmp4658 < -0.5)
      {
        tmp4658 += 1.0;
        tmp4657 -= 1.0;
      }
      if(fabs(tmp4658) < 1e-10 && ((unsigned long)tmp4657 & 1))
      {
        tmp4654 = -pow(-tmp4652, tmp4655)*pow(tmp4652, tmp4656);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4652, tmp4653);
      }
    }
  }
  else
  {
    tmp4654 = pow(tmp4652, tmp4653);
  }
  if(isnan(tmp4654) || isinf(tmp4654))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4652, tmp4653);
  }tmp4659 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4654,"(r_init[35] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4659 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[35] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4659);
    }
  }
  (data->simulationInfo->realParameter[539] /* omega_c[35] PARAM */) = sqrt(tmp4659);
  TRACE_POP
}

/*
equation index: 13933
type: SIMPLE_ASSIGN
r_init[34] = r_min + 34.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13933};
  (data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (34.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13934
type: SIMPLE_ASSIGN
omega_c[34] = sqrt(G * Md / (r_init[34] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13934(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13934};
  modelica_real tmp4660;
  modelica_real tmp4661;
  modelica_real tmp4662;
  modelica_real tmp4663;
  modelica_real tmp4664;
  modelica_real tmp4665;
  modelica_real tmp4666;
  modelica_real tmp4667;
  modelica_real tmp4668;
  modelica_real tmp4669;
  tmp4660 = (data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */);
  tmp4661 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4662 = (tmp4660 * tmp4660) + (tmp4661 * tmp4661);
  tmp4663 = 1.5;
  if(tmp4662 < 0.0 && tmp4663 != 0.0)
  {
    tmp4665 = modf(tmp4663, &tmp4666);
    
    if(tmp4665 > 0.5)
    {
      tmp4665 -= 1.0;
      tmp4666 += 1.0;
    }
    else if(tmp4665 < -0.5)
    {
      tmp4665 += 1.0;
      tmp4666 -= 1.0;
    }
    
    if(fabs(tmp4665) < 1e-10)
      tmp4664 = pow(tmp4662, tmp4666);
    else
    {
      tmp4668 = modf(1.0/tmp4663, &tmp4667);
      if(tmp4668 > 0.5)
      {
        tmp4668 -= 1.0;
        tmp4667 += 1.0;
      }
      else if(tmp4668 < -0.5)
      {
        tmp4668 += 1.0;
        tmp4667 -= 1.0;
      }
      if(fabs(tmp4668) < 1e-10 && ((unsigned long)tmp4667 & 1))
      {
        tmp4664 = -pow(-tmp4662, tmp4665)*pow(tmp4662, tmp4666);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4662, tmp4663);
      }
    }
  }
  else
  {
    tmp4664 = pow(tmp4662, tmp4663);
  }
  if(isnan(tmp4664) || isinf(tmp4664))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4662, tmp4663);
  }tmp4669 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4664,"(r_init[34] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4669 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[34] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4669);
    }
  }
  (data->simulationInfo->realParameter[538] /* omega_c[34] PARAM */) = sqrt(tmp4669);
  TRACE_POP
}

/*
equation index: 13935
type: SIMPLE_ASSIGN
r_init[33] = r_min + 33.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13935};
  (data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (33.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13936
type: SIMPLE_ASSIGN
omega_c[33] = sqrt(G * Md / (r_init[33] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13936(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13936};
  modelica_real tmp4670;
  modelica_real tmp4671;
  modelica_real tmp4672;
  modelica_real tmp4673;
  modelica_real tmp4674;
  modelica_real tmp4675;
  modelica_real tmp4676;
  modelica_real tmp4677;
  modelica_real tmp4678;
  modelica_real tmp4679;
  tmp4670 = (data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */);
  tmp4671 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4672 = (tmp4670 * tmp4670) + (tmp4671 * tmp4671);
  tmp4673 = 1.5;
  if(tmp4672 < 0.0 && tmp4673 != 0.0)
  {
    tmp4675 = modf(tmp4673, &tmp4676);
    
    if(tmp4675 > 0.5)
    {
      tmp4675 -= 1.0;
      tmp4676 += 1.0;
    }
    else if(tmp4675 < -0.5)
    {
      tmp4675 += 1.0;
      tmp4676 -= 1.0;
    }
    
    if(fabs(tmp4675) < 1e-10)
      tmp4674 = pow(tmp4672, tmp4676);
    else
    {
      tmp4678 = modf(1.0/tmp4673, &tmp4677);
      if(tmp4678 > 0.5)
      {
        tmp4678 -= 1.0;
        tmp4677 += 1.0;
      }
      else if(tmp4678 < -0.5)
      {
        tmp4678 += 1.0;
        tmp4677 -= 1.0;
      }
      if(fabs(tmp4678) < 1e-10 && ((unsigned long)tmp4677 & 1))
      {
        tmp4674 = -pow(-tmp4672, tmp4675)*pow(tmp4672, tmp4676);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4672, tmp4673);
      }
    }
  }
  else
  {
    tmp4674 = pow(tmp4672, tmp4673);
  }
  if(isnan(tmp4674) || isinf(tmp4674))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4672, tmp4673);
  }tmp4679 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4674,"(r_init[33] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4679 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[33] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4679);
    }
  }
  (data->simulationInfo->realParameter[537] /* omega_c[33] PARAM */) = sqrt(tmp4679);
  TRACE_POP
}

/*
equation index: 13937
type: SIMPLE_ASSIGN
r_init[32] = r_min + 32.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13937};
  (data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (32.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13938
type: SIMPLE_ASSIGN
omega_c[32] = sqrt(G * Md / (r_init[32] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13938(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13938};
  modelica_real tmp4680;
  modelica_real tmp4681;
  modelica_real tmp4682;
  modelica_real tmp4683;
  modelica_real tmp4684;
  modelica_real tmp4685;
  modelica_real tmp4686;
  modelica_real tmp4687;
  modelica_real tmp4688;
  modelica_real tmp4689;
  tmp4680 = (data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */);
  tmp4681 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4682 = (tmp4680 * tmp4680) + (tmp4681 * tmp4681);
  tmp4683 = 1.5;
  if(tmp4682 < 0.0 && tmp4683 != 0.0)
  {
    tmp4685 = modf(tmp4683, &tmp4686);
    
    if(tmp4685 > 0.5)
    {
      tmp4685 -= 1.0;
      tmp4686 += 1.0;
    }
    else if(tmp4685 < -0.5)
    {
      tmp4685 += 1.0;
      tmp4686 -= 1.0;
    }
    
    if(fabs(tmp4685) < 1e-10)
      tmp4684 = pow(tmp4682, tmp4686);
    else
    {
      tmp4688 = modf(1.0/tmp4683, &tmp4687);
      if(tmp4688 > 0.5)
      {
        tmp4688 -= 1.0;
        tmp4687 += 1.0;
      }
      else if(tmp4688 < -0.5)
      {
        tmp4688 += 1.0;
        tmp4687 -= 1.0;
      }
      if(fabs(tmp4688) < 1e-10 && ((unsigned long)tmp4687 & 1))
      {
        tmp4684 = -pow(-tmp4682, tmp4685)*pow(tmp4682, tmp4686);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4682, tmp4683);
      }
    }
  }
  else
  {
    tmp4684 = pow(tmp4682, tmp4683);
  }
  if(isnan(tmp4684) || isinf(tmp4684))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4682, tmp4683);
  }tmp4689 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4684,"(r_init[32] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4689 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[32] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4689);
    }
  }
  (data->simulationInfo->realParameter[536] /* omega_c[32] PARAM */) = sqrt(tmp4689);
  TRACE_POP
}

/*
equation index: 13939
type: SIMPLE_ASSIGN
r_init[31] = r_min + 31.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13939};
  (data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (31.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13940
type: SIMPLE_ASSIGN
omega_c[31] = sqrt(G * Md / (r_init[31] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13940};
  modelica_real tmp4690;
  modelica_real tmp4691;
  modelica_real tmp4692;
  modelica_real tmp4693;
  modelica_real tmp4694;
  modelica_real tmp4695;
  modelica_real tmp4696;
  modelica_real tmp4697;
  modelica_real tmp4698;
  modelica_real tmp4699;
  tmp4690 = (data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */);
  tmp4691 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4692 = (tmp4690 * tmp4690) + (tmp4691 * tmp4691);
  tmp4693 = 1.5;
  if(tmp4692 < 0.0 && tmp4693 != 0.0)
  {
    tmp4695 = modf(tmp4693, &tmp4696);
    
    if(tmp4695 > 0.5)
    {
      tmp4695 -= 1.0;
      tmp4696 += 1.0;
    }
    else if(tmp4695 < -0.5)
    {
      tmp4695 += 1.0;
      tmp4696 -= 1.0;
    }
    
    if(fabs(tmp4695) < 1e-10)
      tmp4694 = pow(tmp4692, tmp4696);
    else
    {
      tmp4698 = modf(1.0/tmp4693, &tmp4697);
      if(tmp4698 > 0.5)
      {
        tmp4698 -= 1.0;
        tmp4697 += 1.0;
      }
      else if(tmp4698 < -0.5)
      {
        tmp4698 += 1.0;
        tmp4697 -= 1.0;
      }
      if(fabs(tmp4698) < 1e-10 && ((unsigned long)tmp4697 & 1))
      {
        tmp4694 = -pow(-tmp4692, tmp4695)*pow(tmp4692, tmp4696);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4692, tmp4693);
      }
    }
  }
  else
  {
    tmp4694 = pow(tmp4692, tmp4693);
  }
  if(isnan(tmp4694) || isinf(tmp4694))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4692, tmp4693);
  }tmp4699 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4694,"(r_init[31] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4699 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[31] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4699);
    }
  }
  (data->simulationInfo->realParameter[535] /* omega_c[31] PARAM */) = sqrt(tmp4699);
  TRACE_POP
}

/*
equation index: 13941
type: SIMPLE_ASSIGN
r_init[30] = r_min + 30.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13941};
  (data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (30.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13942
type: SIMPLE_ASSIGN
omega_c[30] = sqrt(G * Md / (r_init[30] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13942(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13942};
  modelica_real tmp4700;
  modelica_real tmp4701;
  modelica_real tmp4702;
  modelica_real tmp4703;
  modelica_real tmp4704;
  modelica_real tmp4705;
  modelica_real tmp4706;
  modelica_real tmp4707;
  modelica_real tmp4708;
  modelica_real tmp4709;
  tmp4700 = (data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */);
  tmp4701 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4702 = (tmp4700 * tmp4700) + (tmp4701 * tmp4701);
  tmp4703 = 1.5;
  if(tmp4702 < 0.0 && tmp4703 != 0.0)
  {
    tmp4705 = modf(tmp4703, &tmp4706);
    
    if(tmp4705 > 0.5)
    {
      tmp4705 -= 1.0;
      tmp4706 += 1.0;
    }
    else if(tmp4705 < -0.5)
    {
      tmp4705 += 1.0;
      tmp4706 -= 1.0;
    }
    
    if(fabs(tmp4705) < 1e-10)
      tmp4704 = pow(tmp4702, tmp4706);
    else
    {
      tmp4708 = modf(1.0/tmp4703, &tmp4707);
      if(tmp4708 > 0.5)
      {
        tmp4708 -= 1.0;
        tmp4707 += 1.0;
      }
      else if(tmp4708 < -0.5)
      {
        tmp4708 += 1.0;
        tmp4707 -= 1.0;
      }
      if(fabs(tmp4708) < 1e-10 && ((unsigned long)tmp4707 & 1))
      {
        tmp4704 = -pow(-tmp4702, tmp4705)*pow(tmp4702, tmp4706);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4702, tmp4703);
      }
    }
  }
  else
  {
    tmp4704 = pow(tmp4702, tmp4703);
  }
  if(isnan(tmp4704) || isinf(tmp4704))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4702, tmp4703);
  }tmp4709 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4704,"(r_init[30] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4709 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[30] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4709);
    }
  }
  (data->simulationInfo->realParameter[534] /* omega_c[30] PARAM */) = sqrt(tmp4709);
  TRACE_POP
}

/*
equation index: 13943
type: SIMPLE_ASSIGN
r_init[29] = r_min + 29.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13943};
  (data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (29.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13944
type: SIMPLE_ASSIGN
omega_c[29] = sqrt(G * Md / (r_init[29] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13944(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13944};
  modelica_real tmp4710;
  modelica_real tmp4711;
  modelica_real tmp4712;
  modelica_real tmp4713;
  modelica_real tmp4714;
  modelica_real tmp4715;
  modelica_real tmp4716;
  modelica_real tmp4717;
  modelica_real tmp4718;
  modelica_real tmp4719;
  tmp4710 = (data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */);
  tmp4711 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4712 = (tmp4710 * tmp4710) + (tmp4711 * tmp4711);
  tmp4713 = 1.5;
  if(tmp4712 < 0.0 && tmp4713 != 0.0)
  {
    tmp4715 = modf(tmp4713, &tmp4716);
    
    if(tmp4715 > 0.5)
    {
      tmp4715 -= 1.0;
      tmp4716 += 1.0;
    }
    else if(tmp4715 < -0.5)
    {
      tmp4715 += 1.0;
      tmp4716 -= 1.0;
    }
    
    if(fabs(tmp4715) < 1e-10)
      tmp4714 = pow(tmp4712, tmp4716);
    else
    {
      tmp4718 = modf(1.0/tmp4713, &tmp4717);
      if(tmp4718 > 0.5)
      {
        tmp4718 -= 1.0;
        tmp4717 += 1.0;
      }
      else if(tmp4718 < -0.5)
      {
        tmp4718 += 1.0;
        tmp4717 -= 1.0;
      }
      if(fabs(tmp4718) < 1e-10 && ((unsigned long)tmp4717 & 1))
      {
        tmp4714 = -pow(-tmp4712, tmp4715)*pow(tmp4712, tmp4716);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4712, tmp4713);
      }
    }
  }
  else
  {
    tmp4714 = pow(tmp4712, tmp4713);
  }
  if(isnan(tmp4714) || isinf(tmp4714))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4712, tmp4713);
  }tmp4719 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4714,"(r_init[29] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4719 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[29] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4719);
    }
  }
  (data->simulationInfo->realParameter[533] /* omega_c[29] PARAM */) = sqrt(tmp4719);
  TRACE_POP
}

/*
equation index: 13945
type: SIMPLE_ASSIGN
r_init[28] = r_min + 28.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13945};
  (data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (28.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13946
type: SIMPLE_ASSIGN
omega_c[28] = sqrt(G * Md / (r_init[28] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13946(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13946};
  modelica_real tmp4720;
  modelica_real tmp4721;
  modelica_real tmp4722;
  modelica_real tmp4723;
  modelica_real tmp4724;
  modelica_real tmp4725;
  modelica_real tmp4726;
  modelica_real tmp4727;
  modelica_real tmp4728;
  modelica_real tmp4729;
  tmp4720 = (data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */);
  tmp4721 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4722 = (tmp4720 * tmp4720) + (tmp4721 * tmp4721);
  tmp4723 = 1.5;
  if(tmp4722 < 0.0 && tmp4723 != 0.0)
  {
    tmp4725 = modf(tmp4723, &tmp4726);
    
    if(tmp4725 > 0.5)
    {
      tmp4725 -= 1.0;
      tmp4726 += 1.0;
    }
    else if(tmp4725 < -0.5)
    {
      tmp4725 += 1.0;
      tmp4726 -= 1.0;
    }
    
    if(fabs(tmp4725) < 1e-10)
      tmp4724 = pow(tmp4722, tmp4726);
    else
    {
      tmp4728 = modf(1.0/tmp4723, &tmp4727);
      if(tmp4728 > 0.5)
      {
        tmp4728 -= 1.0;
        tmp4727 += 1.0;
      }
      else if(tmp4728 < -0.5)
      {
        tmp4728 += 1.0;
        tmp4727 -= 1.0;
      }
      if(fabs(tmp4728) < 1e-10 && ((unsigned long)tmp4727 & 1))
      {
        tmp4724 = -pow(-tmp4722, tmp4725)*pow(tmp4722, tmp4726);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4722, tmp4723);
      }
    }
  }
  else
  {
    tmp4724 = pow(tmp4722, tmp4723);
  }
  if(isnan(tmp4724) || isinf(tmp4724))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4722, tmp4723);
  }tmp4729 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4724,"(r_init[28] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4729 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[28] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4729);
    }
  }
  (data->simulationInfo->realParameter[532] /* omega_c[28] PARAM */) = sqrt(tmp4729);
  TRACE_POP
}

/*
equation index: 13947
type: SIMPLE_ASSIGN
r_init[27] = r_min + 27.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13947};
  (data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (27.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13948
type: SIMPLE_ASSIGN
omega_c[27] = sqrt(G * Md / (r_init[27] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13948};
  modelica_real tmp4730;
  modelica_real tmp4731;
  modelica_real tmp4732;
  modelica_real tmp4733;
  modelica_real tmp4734;
  modelica_real tmp4735;
  modelica_real tmp4736;
  modelica_real tmp4737;
  modelica_real tmp4738;
  modelica_real tmp4739;
  tmp4730 = (data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */);
  tmp4731 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4732 = (tmp4730 * tmp4730) + (tmp4731 * tmp4731);
  tmp4733 = 1.5;
  if(tmp4732 < 0.0 && tmp4733 != 0.0)
  {
    tmp4735 = modf(tmp4733, &tmp4736);
    
    if(tmp4735 > 0.5)
    {
      tmp4735 -= 1.0;
      tmp4736 += 1.0;
    }
    else if(tmp4735 < -0.5)
    {
      tmp4735 += 1.0;
      tmp4736 -= 1.0;
    }
    
    if(fabs(tmp4735) < 1e-10)
      tmp4734 = pow(tmp4732, tmp4736);
    else
    {
      tmp4738 = modf(1.0/tmp4733, &tmp4737);
      if(tmp4738 > 0.5)
      {
        tmp4738 -= 1.0;
        tmp4737 += 1.0;
      }
      else if(tmp4738 < -0.5)
      {
        tmp4738 += 1.0;
        tmp4737 -= 1.0;
      }
      if(fabs(tmp4738) < 1e-10 && ((unsigned long)tmp4737 & 1))
      {
        tmp4734 = -pow(-tmp4732, tmp4735)*pow(tmp4732, tmp4736);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4732, tmp4733);
      }
    }
  }
  else
  {
    tmp4734 = pow(tmp4732, tmp4733);
  }
  if(isnan(tmp4734) || isinf(tmp4734))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4732, tmp4733);
  }tmp4739 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4734,"(r_init[27] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4739 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[27] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4739);
    }
  }
  (data->simulationInfo->realParameter[531] /* omega_c[27] PARAM */) = sqrt(tmp4739);
  TRACE_POP
}

/*
equation index: 13949
type: SIMPLE_ASSIGN
r_init[26] = r_min + 26.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13949};
  (data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (26.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13950
type: SIMPLE_ASSIGN
omega_c[26] = sqrt(G * Md / (r_init[26] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13950(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13950};
  modelica_real tmp4740;
  modelica_real tmp4741;
  modelica_real tmp4742;
  modelica_real tmp4743;
  modelica_real tmp4744;
  modelica_real tmp4745;
  modelica_real tmp4746;
  modelica_real tmp4747;
  modelica_real tmp4748;
  modelica_real tmp4749;
  tmp4740 = (data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */);
  tmp4741 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4742 = (tmp4740 * tmp4740) + (tmp4741 * tmp4741);
  tmp4743 = 1.5;
  if(tmp4742 < 0.0 && tmp4743 != 0.0)
  {
    tmp4745 = modf(tmp4743, &tmp4746);
    
    if(tmp4745 > 0.5)
    {
      tmp4745 -= 1.0;
      tmp4746 += 1.0;
    }
    else if(tmp4745 < -0.5)
    {
      tmp4745 += 1.0;
      tmp4746 -= 1.0;
    }
    
    if(fabs(tmp4745) < 1e-10)
      tmp4744 = pow(tmp4742, tmp4746);
    else
    {
      tmp4748 = modf(1.0/tmp4743, &tmp4747);
      if(tmp4748 > 0.5)
      {
        tmp4748 -= 1.0;
        tmp4747 += 1.0;
      }
      else if(tmp4748 < -0.5)
      {
        tmp4748 += 1.0;
        tmp4747 -= 1.0;
      }
      if(fabs(tmp4748) < 1e-10 && ((unsigned long)tmp4747 & 1))
      {
        tmp4744 = -pow(-tmp4742, tmp4745)*pow(tmp4742, tmp4746);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4742, tmp4743);
      }
    }
  }
  else
  {
    tmp4744 = pow(tmp4742, tmp4743);
  }
  if(isnan(tmp4744) || isinf(tmp4744))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4742, tmp4743);
  }tmp4749 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4744,"(r_init[26] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4749 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[26] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4749);
    }
  }
  (data->simulationInfo->realParameter[530] /* omega_c[26] PARAM */) = sqrt(tmp4749);
  TRACE_POP
}

/*
equation index: 13951
type: SIMPLE_ASSIGN
r_init[25] = r_min + 25.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13951};
  (data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (25.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13952
type: SIMPLE_ASSIGN
omega_c[25] = sqrt(G * Md / (r_init[25] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13952(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13952};
  modelica_real tmp4750;
  modelica_real tmp4751;
  modelica_real tmp4752;
  modelica_real tmp4753;
  modelica_real tmp4754;
  modelica_real tmp4755;
  modelica_real tmp4756;
  modelica_real tmp4757;
  modelica_real tmp4758;
  modelica_real tmp4759;
  tmp4750 = (data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */);
  tmp4751 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4752 = (tmp4750 * tmp4750) + (tmp4751 * tmp4751);
  tmp4753 = 1.5;
  if(tmp4752 < 0.0 && tmp4753 != 0.0)
  {
    tmp4755 = modf(tmp4753, &tmp4756);
    
    if(tmp4755 > 0.5)
    {
      tmp4755 -= 1.0;
      tmp4756 += 1.0;
    }
    else if(tmp4755 < -0.5)
    {
      tmp4755 += 1.0;
      tmp4756 -= 1.0;
    }
    
    if(fabs(tmp4755) < 1e-10)
      tmp4754 = pow(tmp4752, tmp4756);
    else
    {
      tmp4758 = modf(1.0/tmp4753, &tmp4757);
      if(tmp4758 > 0.5)
      {
        tmp4758 -= 1.0;
        tmp4757 += 1.0;
      }
      else if(tmp4758 < -0.5)
      {
        tmp4758 += 1.0;
        tmp4757 -= 1.0;
      }
      if(fabs(tmp4758) < 1e-10 && ((unsigned long)tmp4757 & 1))
      {
        tmp4754 = -pow(-tmp4752, tmp4755)*pow(tmp4752, tmp4756);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4752, tmp4753);
      }
    }
  }
  else
  {
    tmp4754 = pow(tmp4752, tmp4753);
  }
  if(isnan(tmp4754) || isinf(tmp4754))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4752, tmp4753);
  }tmp4759 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4754,"(r_init[25] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4759 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[25] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4759);
    }
  }
  (data->simulationInfo->realParameter[529] /* omega_c[25] PARAM */) = sqrt(tmp4759);
  TRACE_POP
}

/*
equation index: 13953
type: SIMPLE_ASSIGN
r_init[24] = r_min + 24.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13953};
  (data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (24.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13954
type: SIMPLE_ASSIGN
omega_c[24] = sqrt(G * Md / (r_init[24] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13954(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13954};
  modelica_real tmp4760;
  modelica_real tmp4761;
  modelica_real tmp4762;
  modelica_real tmp4763;
  modelica_real tmp4764;
  modelica_real tmp4765;
  modelica_real tmp4766;
  modelica_real tmp4767;
  modelica_real tmp4768;
  modelica_real tmp4769;
  tmp4760 = (data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */);
  tmp4761 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4762 = (tmp4760 * tmp4760) + (tmp4761 * tmp4761);
  tmp4763 = 1.5;
  if(tmp4762 < 0.0 && tmp4763 != 0.0)
  {
    tmp4765 = modf(tmp4763, &tmp4766);
    
    if(tmp4765 > 0.5)
    {
      tmp4765 -= 1.0;
      tmp4766 += 1.0;
    }
    else if(tmp4765 < -0.5)
    {
      tmp4765 += 1.0;
      tmp4766 -= 1.0;
    }
    
    if(fabs(tmp4765) < 1e-10)
      tmp4764 = pow(tmp4762, tmp4766);
    else
    {
      tmp4768 = modf(1.0/tmp4763, &tmp4767);
      if(tmp4768 > 0.5)
      {
        tmp4768 -= 1.0;
        tmp4767 += 1.0;
      }
      else if(tmp4768 < -0.5)
      {
        tmp4768 += 1.0;
        tmp4767 -= 1.0;
      }
      if(fabs(tmp4768) < 1e-10 && ((unsigned long)tmp4767 & 1))
      {
        tmp4764 = -pow(-tmp4762, tmp4765)*pow(tmp4762, tmp4766);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4762, tmp4763);
      }
    }
  }
  else
  {
    tmp4764 = pow(tmp4762, tmp4763);
  }
  if(isnan(tmp4764) || isinf(tmp4764))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4762, tmp4763);
  }tmp4769 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4764,"(r_init[24] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4769 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[24] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4769);
    }
  }
  (data->simulationInfo->realParameter[528] /* omega_c[24] PARAM */) = sqrt(tmp4769);
  TRACE_POP
}

/*
equation index: 13955
type: SIMPLE_ASSIGN
r_init[23] = r_min + 23.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13955};
  (data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (23.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13956
type: SIMPLE_ASSIGN
omega_c[23] = sqrt(G * Md / (r_init[23] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13956};
  modelica_real tmp4770;
  modelica_real tmp4771;
  modelica_real tmp4772;
  modelica_real tmp4773;
  modelica_real tmp4774;
  modelica_real tmp4775;
  modelica_real tmp4776;
  modelica_real tmp4777;
  modelica_real tmp4778;
  modelica_real tmp4779;
  tmp4770 = (data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */);
  tmp4771 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4772 = (tmp4770 * tmp4770) + (tmp4771 * tmp4771);
  tmp4773 = 1.5;
  if(tmp4772 < 0.0 && tmp4773 != 0.0)
  {
    tmp4775 = modf(tmp4773, &tmp4776);
    
    if(tmp4775 > 0.5)
    {
      tmp4775 -= 1.0;
      tmp4776 += 1.0;
    }
    else if(tmp4775 < -0.5)
    {
      tmp4775 += 1.0;
      tmp4776 -= 1.0;
    }
    
    if(fabs(tmp4775) < 1e-10)
      tmp4774 = pow(tmp4772, tmp4776);
    else
    {
      tmp4778 = modf(1.0/tmp4773, &tmp4777);
      if(tmp4778 > 0.5)
      {
        tmp4778 -= 1.0;
        tmp4777 += 1.0;
      }
      else if(tmp4778 < -0.5)
      {
        tmp4778 += 1.0;
        tmp4777 -= 1.0;
      }
      if(fabs(tmp4778) < 1e-10 && ((unsigned long)tmp4777 & 1))
      {
        tmp4774 = -pow(-tmp4772, tmp4775)*pow(tmp4772, tmp4776);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4772, tmp4773);
      }
    }
  }
  else
  {
    tmp4774 = pow(tmp4772, tmp4773);
  }
  if(isnan(tmp4774) || isinf(tmp4774))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4772, tmp4773);
  }tmp4779 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4774,"(r_init[23] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4779 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[23] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4779);
    }
  }
  (data->simulationInfo->realParameter[527] /* omega_c[23] PARAM */) = sqrt(tmp4779);
  TRACE_POP
}

/*
equation index: 13957
type: SIMPLE_ASSIGN
r_init[22] = r_min + 22.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13957};
  (data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (22.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13958
type: SIMPLE_ASSIGN
omega_c[22] = sqrt(G * Md / (r_init[22] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13958(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13958};
  modelica_real tmp4780;
  modelica_real tmp4781;
  modelica_real tmp4782;
  modelica_real tmp4783;
  modelica_real tmp4784;
  modelica_real tmp4785;
  modelica_real tmp4786;
  modelica_real tmp4787;
  modelica_real tmp4788;
  modelica_real tmp4789;
  tmp4780 = (data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */);
  tmp4781 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4782 = (tmp4780 * tmp4780) + (tmp4781 * tmp4781);
  tmp4783 = 1.5;
  if(tmp4782 < 0.0 && tmp4783 != 0.0)
  {
    tmp4785 = modf(tmp4783, &tmp4786);
    
    if(tmp4785 > 0.5)
    {
      tmp4785 -= 1.0;
      tmp4786 += 1.0;
    }
    else if(tmp4785 < -0.5)
    {
      tmp4785 += 1.0;
      tmp4786 -= 1.0;
    }
    
    if(fabs(tmp4785) < 1e-10)
      tmp4784 = pow(tmp4782, tmp4786);
    else
    {
      tmp4788 = modf(1.0/tmp4783, &tmp4787);
      if(tmp4788 > 0.5)
      {
        tmp4788 -= 1.0;
        tmp4787 += 1.0;
      }
      else if(tmp4788 < -0.5)
      {
        tmp4788 += 1.0;
        tmp4787 -= 1.0;
      }
      if(fabs(tmp4788) < 1e-10 && ((unsigned long)tmp4787 & 1))
      {
        tmp4784 = -pow(-tmp4782, tmp4785)*pow(tmp4782, tmp4786);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4782, tmp4783);
      }
    }
  }
  else
  {
    tmp4784 = pow(tmp4782, tmp4783);
  }
  if(isnan(tmp4784) || isinf(tmp4784))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4782, tmp4783);
  }tmp4789 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4784,"(r_init[22] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4789 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[22] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4789);
    }
  }
  (data->simulationInfo->realParameter[526] /* omega_c[22] PARAM */) = sqrt(tmp4789);
  TRACE_POP
}

/*
equation index: 13959
type: SIMPLE_ASSIGN
r_init[21] = r_min + 21.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13959};
  (data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (21.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13960
type: SIMPLE_ASSIGN
omega_c[21] = sqrt(G * Md / (r_init[21] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13960(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13960};
  modelica_real tmp4790;
  modelica_real tmp4791;
  modelica_real tmp4792;
  modelica_real tmp4793;
  modelica_real tmp4794;
  modelica_real tmp4795;
  modelica_real tmp4796;
  modelica_real tmp4797;
  modelica_real tmp4798;
  modelica_real tmp4799;
  tmp4790 = (data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */);
  tmp4791 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4792 = (tmp4790 * tmp4790) + (tmp4791 * tmp4791);
  tmp4793 = 1.5;
  if(tmp4792 < 0.0 && tmp4793 != 0.0)
  {
    tmp4795 = modf(tmp4793, &tmp4796);
    
    if(tmp4795 > 0.5)
    {
      tmp4795 -= 1.0;
      tmp4796 += 1.0;
    }
    else if(tmp4795 < -0.5)
    {
      tmp4795 += 1.0;
      tmp4796 -= 1.0;
    }
    
    if(fabs(tmp4795) < 1e-10)
      tmp4794 = pow(tmp4792, tmp4796);
    else
    {
      tmp4798 = modf(1.0/tmp4793, &tmp4797);
      if(tmp4798 > 0.5)
      {
        tmp4798 -= 1.0;
        tmp4797 += 1.0;
      }
      else if(tmp4798 < -0.5)
      {
        tmp4798 += 1.0;
        tmp4797 -= 1.0;
      }
      if(fabs(tmp4798) < 1e-10 && ((unsigned long)tmp4797 & 1))
      {
        tmp4794 = -pow(-tmp4792, tmp4795)*pow(tmp4792, tmp4796);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4792, tmp4793);
      }
    }
  }
  else
  {
    tmp4794 = pow(tmp4792, tmp4793);
  }
  if(isnan(tmp4794) || isinf(tmp4794))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4792, tmp4793);
  }tmp4799 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4794,"(r_init[21] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4799 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[21] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4799);
    }
  }
  (data->simulationInfo->realParameter[525] /* omega_c[21] PARAM */) = sqrt(tmp4799);
  TRACE_POP
}

/*
equation index: 13961
type: SIMPLE_ASSIGN
r_init[20] = r_min + 20.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13961};
  (data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (20.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13962
type: SIMPLE_ASSIGN
omega_c[20] = sqrt(G * Md / (r_init[20] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13962(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13962};
  modelica_real tmp4800;
  modelica_real tmp4801;
  modelica_real tmp4802;
  modelica_real tmp4803;
  modelica_real tmp4804;
  modelica_real tmp4805;
  modelica_real tmp4806;
  modelica_real tmp4807;
  modelica_real tmp4808;
  modelica_real tmp4809;
  tmp4800 = (data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */);
  tmp4801 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4802 = (tmp4800 * tmp4800) + (tmp4801 * tmp4801);
  tmp4803 = 1.5;
  if(tmp4802 < 0.0 && tmp4803 != 0.0)
  {
    tmp4805 = modf(tmp4803, &tmp4806);
    
    if(tmp4805 > 0.5)
    {
      tmp4805 -= 1.0;
      tmp4806 += 1.0;
    }
    else if(tmp4805 < -0.5)
    {
      tmp4805 += 1.0;
      tmp4806 -= 1.0;
    }
    
    if(fabs(tmp4805) < 1e-10)
      tmp4804 = pow(tmp4802, tmp4806);
    else
    {
      tmp4808 = modf(1.0/tmp4803, &tmp4807);
      if(tmp4808 > 0.5)
      {
        tmp4808 -= 1.0;
        tmp4807 += 1.0;
      }
      else if(tmp4808 < -0.5)
      {
        tmp4808 += 1.0;
        tmp4807 -= 1.0;
      }
      if(fabs(tmp4808) < 1e-10 && ((unsigned long)tmp4807 & 1))
      {
        tmp4804 = -pow(-tmp4802, tmp4805)*pow(tmp4802, tmp4806);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4802, tmp4803);
      }
    }
  }
  else
  {
    tmp4804 = pow(tmp4802, tmp4803);
  }
  if(isnan(tmp4804) || isinf(tmp4804))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4802, tmp4803);
  }tmp4809 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4804,"(r_init[20] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4809 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[20] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4809);
    }
  }
  (data->simulationInfo->realParameter[524] /* omega_c[20] PARAM */) = sqrt(tmp4809);
  TRACE_POP
}

/*
equation index: 13963
type: SIMPLE_ASSIGN
r_init[19] = r_min + 19.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13963};
  (data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (19.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13964
type: SIMPLE_ASSIGN
omega_c[19] = sqrt(G * Md / (r_init[19] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13964};
  modelica_real tmp4810;
  modelica_real tmp4811;
  modelica_real tmp4812;
  modelica_real tmp4813;
  modelica_real tmp4814;
  modelica_real tmp4815;
  modelica_real tmp4816;
  modelica_real tmp4817;
  modelica_real tmp4818;
  modelica_real tmp4819;
  tmp4810 = (data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */);
  tmp4811 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4812 = (tmp4810 * tmp4810) + (tmp4811 * tmp4811);
  tmp4813 = 1.5;
  if(tmp4812 < 0.0 && tmp4813 != 0.0)
  {
    tmp4815 = modf(tmp4813, &tmp4816);
    
    if(tmp4815 > 0.5)
    {
      tmp4815 -= 1.0;
      tmp4816 += 1.0;
    }
    else if(tmp4815 < -0.5)
    {
      tmp4815 += 1.0;
      tmp4816 -= 1.0;
    }
    
    if(fabs(tmp4815) < 1e-10)
      tmp4814 = pow(tmp4812, tmp4816);
    else
    {
      tmp4818 = modf(1.0/tmp4813, &tmp4817);
      if(tmp4818 > 0.5)
      {
        tmp4818 -= 1.0;
        tmp4817 += 1.0;
      }
      else if(tmp4818 < -0.5)
      {
        tmp4818 += 1.0;
        tmp4817 -= 1.0;
      }
      if(fabs(tmp4818) < 1e-10 && ((unsigned long)tmp4817 & 1))
      {
        tmp4814 = -pow(-tmp4812, tmp4815)*pow(tmp4812, tmp4816);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4812, tmp4813);
      }
    }
  }
  else
  {
    tmp4814 = pow(tmp4812, tmp4813);
  }
  if(isnan(tmp4814) || isinf(tmp4814))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4812, tmp4813);
  }tmp4819 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4814,"(r_init[19] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4819 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[19] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4819);
    }
  }
  (data->simulationInfo->realParameter[523] /* omega_c[19] PARAM */) = sqrt(tmp4819);
  TRACE_POP
}

/*
equation index: 13965
type: SIMPLE_ASSIGN
r_init[18] = r_min + 18.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13965};
  (data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (18.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13966
type: SIMPLE_ASSIGN
omega_c[18] = sqrt(G * Md / (r_init[18] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13966(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13966};
  modelica_real tmp4820;
  modelica_real tmp4821;
  modelica_real tmp4822;
  modelica_real tmp4823;
  modelica_real tmp4824;
  modelica_real tmp4825;
  modelica_real tmp4826;
  modelica_real tmp4827;
  modelica_real tmp4828;
  modelica_real tmp4829;
  tmp4820 = (data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */);
  tmp4821 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4822 = (tmp4820 * tmp4820) + (tmp4821 * tmp4821);
  tmp4823 = 1.5;
  if(tmp4822 < 0.0 && tmp4823 != 0.0)
  {
    tmp4825 = modf(tmp4823, &tmp4826);
    
    if(tmp4825 > 0.5)
    {
      tmp4825 -= 1.0;
      tmp4826 += 1.0;
    }
    else if(tmp4825 < -0.5)
    {
      tmp4825 += 1.0;
      tmp4826 -= 1.0;
    }
    
    if(fabs(tmp4825) < 1e-10)
      tmp4824 = pow(tmp4822, tmp4826);
    else
    {
      tmp4828 = modf(1.0/tmp4823, &tmp4827);
      if(tmp4828 > 0.5)
      {
        tmp4828 -= 1.0;
        tmp4827 += 1.0;
      }
      else if(tmp4828 < -0.5)
      {
        tmp4828 += 1.0;
        tmp4827 -= 1.0;
      }
      if(fabs(tmp4828) < 1e-10 && ((unsigned long)tmp4827 & 1))
      {
        tmp4824 = -pow(-tmp4822, tmp4825)*pow(tmp4822, tmp4826);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4822, tmp4823);
      }
    }
  }
  else
  {
    tmp4824 = pow(tmp4822, tmp4823);
  }
  if(isnan(tmp4824) || isinf(tmp4824))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4822, tmp4823);
  }tmp4829 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4824,"(r_init[18] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4829 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[18] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4829);
    }
  }
  (data->simulationInfo->realParameter[522] /* omega_c[18] PARAM */) = sqrt(tmp4829);
  TRACE_POP
}

/*
equation index: 13967
type: SIMPLE_ASSIGN
r_init[17] = r_min + 17.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13967};
  (data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (17.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13968
type: SIMPLE_ASSIGN
omega_c[17] = sqrt(G * Md / (r_init[17] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13968(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13968};
  modelica_real tmp4830;
  modelica_real tmp4831;
  modelica_real tmp4832;
  modelica_real tmp4833;
  modelica_real tmp4834;
  modelica_real tmp4835;
  modelica_real tmp4836;
  modelica_real tmp4837;
  modelica_real tmp4838;
  modelica_real tmp4839;
  tmp4830 = (data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */);
  tmp4831 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4832 = (tmp4830 * tmp4830) + (tmp4831 * tmp4831);
  tmp4833 = 1.5;
  if(tmp4832 < 0.0 && tmp4833 != 0.0)
  {
    tmp4835 = modf(tmp4833, &tmp4836);
    
    if(tmp4835 > 0.5)
    {
      tmp4835 -= 1.0;
      tmp4836 += 1.0;
    }
    else if(tmp4835 < -0.5)
    {
      tmp4835 += 1.0;
      tmp4836 -= 1.0;
    }
    
    if(fabs(tmp4835) < 1e-10)
      tmp4834 = pow(tmp4832, tmp4836);
    else
    {
      tmp4838 = modf(1.0/tmp4833, &tmp4837);
      if(tmp4838 > 0.5)
      {
        tmp4838 -= 1.0;
        tmp4837 += 1.0;
      }
      else if(tmp4838 < -0.5)
      {
        tmp4838 += 1.0;
        tmp4837 -= 1.0;
      }
      if(fabs(tmp4838) < 1e-10 && ((unsigned long)tmp4837 & 1))
      {
        tmp4834 = -pow(-tmp4832, tmp4835)*pow(tmp4832, tmp4836);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4832, tmp4833);
      }
    }
  }
  else
  {
    tmp4834 = pow(tmp4832, tmp4833);
  }
  if(isnan(tmp4834) || isinf(tmp4834))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4832, tmp4833);
  }tmp4839 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4834,"(r_init[17] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4839 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[17] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4839);
    }
  }
  (data->simulationInfo->realParameter[521] /* omega_c[17] PARAM */) = sqrt(tmp4839);
  TRACE_POP
}

/*
equation index: 13969
type: SIMPLE_ASSIGN
r_init[16] = r_min + 16.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13969};
  (data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (16.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13970
type: SIMPLE_ASSIGN
omega_c[16] = sqrt(G * Md / (r_init[16] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13970(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13970};
  modelica_real tmp4840;
  modelica_real tmp4841;
  modelica_real tmp4842;
  modelica_real tmp4843;
  modelica_real tmp4844;
  modelica_real tmp4845;
  modelica_real tmp4846;
  modelica_real tmp4847;
  modelica_real tmp4848;
  modelica_real tmp4849;
  tmp4840 = (data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */);
  tmp4841 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4842 = (tmp4840 * tmp4840) + (tmp4841 * tmp4841);
  tmp4843 = 1.5;
  if(tmp4842 < 0.0 && tmp4843 != 0.0)
  {
    tmp4845 = modf(tmp4843, &tmp4846);
    
    if(tmp4845 > 0.5)
    {
      tmp4845 -= 1.0;
      tmp4846 += 1.0;
    }
    else if(tmp4845 < -0.5)
    {
      tmp4845 += 1.0;
      tmp4846 -= 1.0;
    }
    
    if(fabs(tmp4845) < 1e-10)
      tmp4844 = pow(tmp4842, tmp4846);
    else
    {
      tmp4848 = modf(1.0/tmp4843, &tmp4847);
      if(tmp4848 > 0.5)
      {
        tmp4848 -= 1.0;
        tmp4847 += 1.0;
      }
      else if(tmp4848 < -0.5)
      {
        tmp4848 += 1.0;
        tmp4847 -= 1.0;
      }
      if(fabs(tmp4848) < 1e-10 && ((unsigned long)tmp4847 & 1))
      {
        tmp4844 = -pow(-tmp4842, tmp4845)*pow(tmp4842, tmp4846);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4842, tmp4843);
      }
    }
  }
  else
  {
    tmp4844 = pow(tmp4842, tmp4843);
  }
  if(isnan(tmp4844) || isinf(tmp4844))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4842, tmp4843);
  }tmp4849 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4844,"(r_init[16] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4849 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[16] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4849);
    }
  }
  (data->simulationInfo->realParameter[520] /* omega_c[16] PARAM */) = sqrt(tmp4849);
  TRACE_POP
}

/*
equation index: 13971
type: SIMPLE_ASSIGN
r_init[15] = r_min + 15.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13971};
  (data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (15.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13972
type: SIMPLE_ASSIGN
omega_c[15] = sqrt(G * Md / (r_init[15] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13972};
  modelica_real tmp4850;
  modelica_real tmp4851;
  modelica_real tmp4852;
  modelica_real tmp4853;
  modelica_real tmp4854;
  modelica_real tmp4855;
  modelica_real tmp4856;
  modelica_real tmp4857;
  modelica_real tmp4858;
  modelica_real tmp4859;
  tmp4850 = (data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */);
  tmp4851 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4852 = (tmp4850 * tmp4850) + (tmp4851 * tmp4851);
  tmp4853 = 1.5;
  if(tmp4852 < 0.0 && tmp4853 != 0.0)
  {
    tmp4855 = modf(tmp4853, &tmp4856);
    
    if(tmp4855 > 0.5)
    {
      tmp4855 -= 1.0;
      tmp4856 += 1.0;
    }
    else if(tmp4855 < -0.5)
    {
      tmp4855 += 1.0;
      tmp4856 -= 1.0;
    }
    
    if(fabs(tmp4855) < 1e-10)
      tmp4854 = pow(tmp4852, tmp4856);
    else
    {
      tmp4858 = modf(1.0/tmp4853, &tmp4857);
      if(tmp4858 > 0.5)
      {
        tmp4858 -= 1.0;
        tmp4857 += 1.0;
      }
      else if(tmp4858 < -0.5)
      {
        tmp4858 += 1.0;
        tmp4857 -= 1.0;
      }
      if(fabs(tmp4858) < 1e-10 && ((unsigned long)tmp4857 & 1))
      {
        tmp4854 = -pow(-tmp4852, tmp4855)*pow(tmp4852, tmp4856);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4852, tmp4853);
      }
    }
  }
  else
  {
    tmp4854 = pow(tmp4852, tmp4853);
  }
  if(isnan(tmp4854) || isinf(tmp4854))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4852, tmp4853);
  }tmp4859 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4854,"(r_init[15] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4859 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[15] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4859);
    }
  }
  (data->simulationInfo->realParameter[519] /* omega_c[15] PARAM */) = sqrt(tmp4859);
  TRACE_POP
}

/*
equation index: 13973
type: SIMPLE_ASSIGN
r_init[14] = r_min + 14.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13973};
  (data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (14.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13974
type: SIMPLE_ASSIGN
omega_c[14] = sqrt(G * Md / (r_init[14] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13974(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13974};
  modelica_real tmp4860;
  modelica_real tmp4861;
  modelica_real tmp4862;
  modelica_real tmp4863;
  modelica_real tmp4864;
  modelica_real tmp4865;
  modelica_real tmp4866;
  modelica_real tmp4867;
  modelica_real tmp4868;
  modelica_real tmp4869;
  tmp4860 = (data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */);
  tmp4861 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4862 = (tmp4860 * tmp4860) + (tmp4861 * tmp4861);
  tmp4863 = 1.5;
  if(tmp4862 < 0.0 && tmp4863 != 0.0)
  {
    tmp4865 = modf(tmp4863, &tmp4866);
    
    if(tmp4865 > 0.5)
    {
      tmp4865 -= 1.0;
      tmp4866 += 1.0;
    }
    else if(tmp4865 < -0.5)
    {
      tmp4865 += 1.0;
      tmp4866 -= 1.0;
    }
    
    if(fabs(tmp4865) < 1e-10)
      tmp4864 = pow(tmp4862, tmp4866);
    else
    {
      tmp4868 = modf(1.0/tmp4863, &tmp4867);
      if(tmp4868 > 0.5)
      {
        tmp4868 -= 1.0;
        tmp4867 += 1.0;
      }
      else if(tmp4868 < -0.5)
      {
        tmp4868 += 1.0;
        tmp4867 -= 1.0;
      }
      if(fabs(tmp4868) < 1e-10 && ((unsigned long)tmp4867 & 1))
      {
        tmp4864 = -pow(-tmp4862, tmp4865)*pow(tmp4862, tmp4866);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4862, tmp4863);
      }
    }
  }
  else
  {
    tmp4864 = pow(tmp4862, tmp4863);
  }
  if(isnan(tmp4864) || isinf(tmp4864))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4862, tmp4863);
  }tmp4869 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4864,"(r_init[14] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4869 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[14] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4869);
    }
  }
  (data->simulationInfo->realParameter[518] /* omega_c[14] PARAM */) = sqrt(tmp4869);
  TRACE_POP
}

/*
equation index: 13975
type: SIMPLE_ASSIGN
r_init[13] = r_min + 13.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13975};
  (data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (13.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13976
type: SIMPLE_ASSIGN
omega_c[13] = sqrt(G * Md / (r_init[13] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13976(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13976};
  modelica_real tmp4870;
  modelica_real tmp4871;
  modelica_real tmp4872;
  modelica_real tmp4873;
  modelica_real tmp4874;
  modelica_real tmp4875;
  modelica_real tmp4876;
  modelica_real tmp4877;
  modelica_real tmp4878;
  modelica_real tmp4879;
  tmp4870 = (data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */);
  tmp4871 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4872 = (tmp4870 * tmp4870) + (tmp4871 * tmp4871);
  tmp4873 = 1.5;
  if(tmp4872 < 0.0 && tmp4873 != 0.0)
  {
    tmp4875 = modf(tmp4873, &tmp4876);
    
    if(tmp4875 > 0.5)
    {
      tmp4875 -= 1.0;
      tmp4876 += 1.0;
    }
    else if(tmp4875 < -0.5)
    {
      tmp4875 += 1.0;
      tmp4876 -= 1.0;
    }
    
    if(fabs(tmp4875) < 1e-10)
      tmp4874 = pow(tmp4872, tmp4876);
    else
    {
      tmp4878 = modf(1.0/tmp4873, &tmp4877);
      if(tmp4878 > 0.5)
      {
        tmp4878 -= 1.0;
        tmp4877 += 1.0;
      }
      else if(tmp4878 < -0.5)
      {
        tmp4878 += 1.0;
        tmp4877 -= 1.0;
      }
      if(fabs(tmp4878) < 1e-10 && ((unsigned long)tmp4877 & 1))
      {
        tmp4874 = -pow(-tmp4872, tmp4875)*pow(tmp4872, tmp4876);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4872, tmp4873);
      }
    }
  }
  else
  {
    tmp4874 = pow(tmp4872, tmp4873);
  }
  if(isnan(tmp4874) || isinf(tmp4874))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4872, tmp4873);
  }tmp4879 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4874,"(r_init[13] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4879 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[13] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4879);
    }
  }
  (data->simulationInfo->realParameter[517] /* omega_c[13] PARAM */) = sqrt(tmp4879);
  TRACE_POP
}

/*
equation index: 13977
type: SIMPLE_ASSIGN
r_init[12] = r_min + 12.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13977};
  (data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (12.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13978
type: SIMPLE_ASSIGN
omega_c[12] = sqrt(G * Md / (r_init[12] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13978(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13978};
  modelica_real tmp4880;
  modelica_real tmp4881;
  modelica_real tmp4882;
  modelica_real tmp4883;
  modelica_real tmp4884;
  modelica_real tmp4885;
  modelica_real tmp4886;
  modelica_real tmp4887;
  modelica_real tmp4888;
  modelica_real tmp4889;
  tmp4880 = (data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */);
  tmp4881 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4882 = (tmp4880 * tmp4880) + (tmp4881 * tmp4881);
  tmp4883 = 1.5;
  if(tmp4882 < 0.0 && tmp4883 != 0.0)
  {
    tmp4885 = modf(tmp4883, &tmp4886);
    
    if(tmp4885 > 0.5)
    {
      tmp4885 -= 1.0;
      tmp4886 += 1.0;
    }
    else if(tmp4885 < -0.5)
    {
      tmp4885 += 1.0;
      tmp4886 -= 1.0;
    }
    
    if(fabs(tmp4885) < 1e-10)
      tmp4884 = pow(tmp4882, tmp4886);
    else
    {
      tmp4888 = modf(1.0/tmp4883, &tmp4887);
      if(tmp4888 > 0.5)
      {
        tmp4888 -= 1.0;
        tmp4887 += 1.0;
      }
      else if(tmp4888 < -0.5)
      {
        tmp4888 += 1.0;
        tmp4887 -= 1.0;
      }
      if(fabs(tmp4888) < 1e-10 && ((unsigned long)tmp4887 & 1))
      {
        tmp4884 = -pow(-tmp4882, tmp4885)*pow(tmp4882, tmp4886);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4882, tmp4883);
      }
    }
  }
  else
  {
    tmp4884 = pow(tmp4882, tmp4883);
  }
  if(isnan(tmp4884) || isinf(tmp4884))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4882, tmp4883);
  }tmp4889 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4884,"(r_init[12] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4889 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[12] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4889);
    }
  }
  (data->simulationInfo->realParameter[516] /* omega_c[12] PARAM */) = sqrt(tmp4889);
  TRACE_POP
}

/*
equation index: 13979
type: SIMPLE_ASSIGN
r_init[11] = r_min + 11.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13979};
  (data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (11.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13980
type: SIMPLE_ASSIGN
omega_c[11] = sqrt(G * Md / (r_init[11] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13980};
  modelica_real tmp4890;
  modelica_real tmp4891;
  modelica_real tmp4892;
  modelica_real tmp4893;
  modelica_real tmp4894;
  modelica_real tmp4895;
  modelica_real tmp4896;
  modelica_real tmp4897;
  modelica_real tmp4898;
  modelica_real tmp4899;
  tmp4890 = (data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */);
  tmp4891 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4892 = (tmp4890 * tmp4890) + (tmp4891 * tmp4891);
  tmp4893 = 1.5;
  if(tmp4892 < 0.0 && tmp4893 != 0.0)
  {
    tmp4895 = modf(tmp4893, &tmp4896);
    
    if(tmp4895 > 0.5)
    {
      tmp4895 -= 1.0;
      tmp4896 += 1.0;
    }
    else if(tmp4895 < -0.5)
    {
      tmp4895 += 1.0;
      tmp4896 -= 1.0;
    }
    
    if(fabs(tmp4895) < 1e-10)
      tmp4894 = pow(tmp4892, tmp4896);
    else
    {
      tmp4898 = modf(1.0/tmp4893, &tmp4897);
      if(tmp4898 > 0.5)
      {
        tmp4898 -= 1.0;
        tmp4897 += 1.0;
      }
      else if(tmp4898 < -0.5)
      {
        tmp4898 += 1.0;
        tmp4897 -= 1.0;
      }
      if(fabs(tmp4898) < 1e-10 && ((unsigned long)tmp4897 & 1))
      {
        tmp4894 = -pow(-tmp4892, tmp4895)*pow(tmp4892, tmp4896);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4892, tmp4893);
      }
    }
  }
  else
  {
    tmp4894 = pow(tmp4892, tmp4893);
  }
  if(isnan(tmp4894) || isinf(tmp4894))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4892, tmp4893);
  }tmp4899 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4894,"(r_init[11] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4899 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[11] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4899);
    }
  }
  (data->simulationInfo->realParameter[515] /* omega_c[11] PARAM */) = sqrt(tmp4899);
  TRACE_POP
}

/*
equation index: 13981
type: SIMPLE_ASSIGN
r_init[10] = r_min + 10.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13981};
  (data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (10.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13982
type: SIMPLE_ASSIGN
omega_c[10] = sqrt(G * Md / (r_init[10] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13982(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13982};
  modelica_real tmp4900;
  modelica_real tmp4901;
  modelica_real tmp4902;
  modelica_real tmp4903;
  modelica_real tmp4904;
  modelica_real tmp4905;
  modelica_real tmp4906;
  modelica_real tmp4907;
  modelica_real tmp4908;
  modelica_real tmp4909;
  tmp4900 = (data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */);
  tmp4901 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4902 = (tmp4900 * tmp4900) + (tmp4901 * tmp4901);
  tmp4903 = 1.5;
  if(tmp4902 < 0.0 && tmp4903 != 0.0)
  {
    tmp4905 = modf(tmp4903, &tmp4906);
    
    if(tmp4905 > 0.5)
    {
      tmp4905 -= 1.0;
      tmp4906 += 1.0;
    }
    else if(tmp4905 < -0.5)
    {
      tmp4905 += 1.0;
      tmp4906 -= 1.0;
    }
    
    if(fabs(tmp4905) < 1e-10)
      tmp4904 = pow(tmp4902, tmp4906);
    else
    {
      tmp4908 = modf(1.0/tmp4903, &tmp4907);
      if(tmp4908 > 0.5)
      {
        tmp4908 -= 1.0;
        tmp4907 += 1.0;
      }
      else if(tmp4908 < -0.5)
      {
        tmp4908 += 1.0;
        tmp4907 -= 1.0;
      }
      if(fabs(tmp4908) < 1e-10 && ((unsigned long)tmp4907 & 1))
      {
        tmp4904 = -pow(-tmp4902, tmp4905)*pow(tmp4902, tmp4906);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4902, tmp4903);
      }
    }
  }
  else
  {
    tmp4904 = pow(tmp4902, tmp4903);
  }
  if(isnan(tmp4904) || isinf(tmp4904))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4902, tmp4903);
  }tmp4909 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4904,"(r_init[10] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4909 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[10] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4909);
    }
  }
  (data->simulationInfo->realParameter[514] /* omega_c[10] PARAM */) = sqrt(tmp4909);
  TRACE_POP
}

/*
equation index: 13983
type: SIMPLE_ASSIGN
r_init[9] = r_min + 9.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13983};
  (data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (9.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13984
type: SIMPLE_ASSIGN
omega_c[9] = sqrt(G * Md / (r_init[9] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13984(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13984};
  modelica_real tmp4910;
  modelica_real tmp4911;
  modelica_real tmp4912;
  modelica_real tmp4913;
  modelica_real tmp4914;
  modelica_real tmp4915;
  modelica_real tmp4916;
  modelica_real tmp4917;
  modelica_real tmp4918;
  modelica_real tmp4919;
  tmp4910 = (data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */);
  tmp4911 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4912 = (tmp4910 * tmp4910) + (tmp4911 * tmp4911);
  tmp4913 = 1.5;
  if(tmp4912 < 0.0 && tmp4913 != 0.0)
  {
    tmp4915 = modf(tmp4913, &tmp4916);
    
    if(tmp4915 > 0.5)
    {
      tmp4915 -= 1.0;
      tmp4916 += 1.0;
    }
    else if(tmp4915 < -0.5)
    {
      tmp4915 += 1.0;
      tmp4916 -= 1.0;
    }
    
    if(fabs(tmp4915) < 1e-10)
      tmp4914 = pow(tmp4912, tmp4916);
    else
    {
      tmp4918 = modf(1.0/tmp4913, &tmp4917);
      if(tmp4918 > 0.5)
      {
        tmp4918 -= 1.0;
        tmp4917 += 1.0;
      }
      else if(tmp4918 < -0.5)
      {
        tmp4918 += 1.0;
        tmp4917 -= 1.0;
      }
      if(fabs(tmp4918) < 1e-10 && ((unsigned long)tmp4917 & 1))
      {
        tmp4914 = -pow(-tmp4912, tmp4915)*pow(tmp4912, tmp4916);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4912, tmp4913);
      }
    }
  }
  else
  {
    tmp4914 = pow(tmp4912, tmp4913);
  }
  if(isnan(tmp4914) || isinf(tmp4914))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4912, tmp4913);
  }tmp4919 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4914,"(r_init[9] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4919 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[9] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4919);
    }
  }
  (data->simulationInfo->realParameter[513] /* omega_c[9] PARAM */) = sqrt(tmp4919);
  TRACE_POP
}

/*
equation index: 13985
type: SIMPLE_ASSIGN
r_init[8] = r_min + 8.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13985};
  (data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (8.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13986
type: SIMPLE_ASSIGN
omega_c[8] = sqrt(G * Md / (r_init[8] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13986(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13986};
  modelica_real tmp4920;
  modelica_real tmp4921;
  modelica_real tmp4922;
  modelica_real tmp4923;
  modelica_real tmp4924;
  modelica_real tmp4925;
  modelica_real tmp4926;
  modelica_real tmp4927;
  modelica_real tmp4928;
  modelica_real tmp4929;
  tmp4920 = (data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */);
  tmp4921 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4922 = (tmp4920 * tmp4920) + (tmp4921 * tmp4921);
  tmp4923 = 1.5;
  if(tmp4922 < 0.0 && tmp4923 != 0.0)
  {
    tmp4925 = modf(tmp4923, &tmp4926);
    
    if(tmp4925 > 0.5)
    {
      tmp4925 -= 1.0;
      tmp4926 += 1.0;
    }
    else if(tmp4925 < -0.5)
    {
      tmp4925 += 1.0;
      tmp4926 -= 1.0;
    }
    
    if(fabs(tmp4925) < 1e-10)
      tmp4924 = pow(tmp4922, tmp4926);
    else
    {
      tmp4928 = modf(1.0/tmp4923, &tmp4927);
      if(tmp4928 > 0.5)
      {
        tmp4928 -= 1.0;
        tmp4927 += 1.0;
      }
      else if(tmp4928 < -0.5)
      {
        tmp4928 += 1.0;
        tmp4927 -= 1.0;
      }
      if(fabs(tmp4928) < 1e-10 && ((unsigned long)tmp4927 & 1))
      {
        tmp4924 = -pow(-tmp4922, tmp4925)*pow(tmp4922, tmp4926);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4922, tmp4923);
      }
    }
  }
  else
  {
    tmp4924 = pow(tmp4922, tmp4923);
  }
  if(isnan(tmp4924) || isinf(tmp4924))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4922, tmp4923);
  }tmp4929 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4924,"(r_init[8] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4929 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[8] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4929);
    }
  }
  (data->simulationInfo->realParameter[512] /* omega_c[8] PARAM */) = sqrt(tmp4929);
  TRACE_POP
}

/*
equation index: 13987
type: SIMPLE_ASSIGN
r_init[7] = r_min + 7.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13987};
  (data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (7.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13988
type: SIMPLE_ASSIGN
omega_c[7] = sqrt(G * Md / (r_init[7] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13988};
  modelica_real tmp4930;
  modelica_real tmp4931;
  modelica_real tmp4932;
  modelica_real tmp4933;
  modelica_real tmp4934;
  modelica_real tmp4935;
  modelica_real tmp4936;
  modelica_real tmp4937;
  modelica_real tmp4938;
  modelica_real tmp4939;
  tmp4930 = (data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */);
  tmp4931 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4932 = (tmp4930 * tmp4930) + (tmp4931 * tmp4931);
  tmp4933 = 1.5;
  if(tmp4932 < 0.0 && tmp4933 != 0.0)
  {
    tmp4935 = modf(tmp4933, &tmp4936);
    
    if(tmp4935 > 0.5)
    {
      tmp4935 -= 1.0;
      tmp4936 += 1.0;
    }
    else if(tmp4935 < -0.5)
    {
      tmp4935 += 1.0;
      tmp4936 -= 1.0;
    }
    
    if(fabs(tmp4935) < 1e-10)
      tmp4934 = pow(tmp4932, tmp4936);
    else
    {
      tmp4938 = modf(1.0/tmp4933, &tmp4937);
      if(tmp4938 > 0.5)
      {
        tmp4938 -= 1.0;
        tmp4937 += 1.0;
      }
      else if(tmp4938 < -0.5)
      {
        tmp4938 += 1.0;
        tmp4937 -= 1.0;
      }
      if(fabs(tmp4938) < 1e-10 && ((unsigned long)tmp4937 & 1))
      {
        tmp4934 = -pow(-tmp4932, tmp4935)*pow(tmp4932, tmp4936);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4932, tmp4933);
      }
    }
  }
  else
  {
    tmp4934 = pow(tmp4932, tmp4933);
  }
  if(isnan(tmp4934) || isinf(tmp4934))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4932, tmp4933);
  }tmp4939 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4934,"(r_init[7] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4939 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[7] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4939);
    }
  }
  (data->simulationInfo->realParameter[511] /* omega_c[7] PARAM */) = sqrt(tmp4939);
  TRACE_POP
}

/*
equation index: 13989
type: SIMPLE_ASSIGN
r_init[6] = r_min + 6.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13989};
  (data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (6.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13990
type: SIMPLE_ASSIGN
omega_c[6] = sqrt(G * Md / (r_init[6] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13990(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13990};
  modelica_real tmp4940;
  modelica_real tmp4941;
  modelica_real tmp4942;
  modelica_real tmp4943;
  modelica_real tmp4944;
  modelica_real tmp4945;
  modelica_real tmp4946;
  modelica_real tmp4947;
  modelica_real tmp4948;
  modelica_real tmp4949;
  tmp4940 = (data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */);
  tmp4941 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4942 = (tmp4940 * tmp4940) + (tmp4941 * tmp4941);
  tmp4943 = 1.5;
  if(tmp4942 < 0.0 && tmp4943 != 0.0)
  {
    tmp4945 = modf(tmp4943, &tmp4946);
    
    if(tmp4945 > 0.5)
    {
      tmp4945 -= 1.0;
      tmp4946 += 1.0;
    }
    else if(tmp4945 < -0.5)
    {
      tmp4945 += 1.0;
      tmp4946 -= 1.0;
    }
    
    if(fabs(tmp4945) < 1e-10)
      tmp4944 = pow(tmp4942, tmp4946);
    else
    {
      tmp4948 = modf(1.0/tmp4943, &tmp4947);
      if(tmp4948 > 0.5)
      {
        tmp4948 -= 1.0;
        tmp4947 += 1.0;
      }
      else if(tmp4948 < -0.5)
      {
        tmp4948 += 1.0;
        tmp4947 -= 1.0;
      }
      if(fabs(tmp4948) < 1e-10 && ((unsigned long)tmp4947 & 1))
      {
        tmp4944 = -pow(-tmp4942, tmp4945)*pow(tmp4942, tmp4946);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4942, tmp4943);
      }
    }
  }
  else
  {
    tmp4944 = pow(tmp4942, tmp4943);
  }
  if(isnan(tmp4944) || isinf(tmp4944))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4942, tmp4943);
  }tmp4949 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4944,"(r_init[6] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4949 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[6] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4949);
    }
  }
  (data->simulationInfo->realParameter[510] /* omega_c[6] PARAM */) = sqrt(tmp4949);
  TRACE_POP
}

/*
equation index: 13991
type: SIMPLE_ASSIGN
r_init[5] = r_min + 5.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13991};
  (data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (5.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13992
type: SIMPLE_ASSIGN
omega_c[5] = sqrt(G * Md / (r_init[5] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13992(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13992};
  modelica_real tmp4950;
  modelica_real tmp4951;
  modelica_real tmp4952;
  modelica_real tmp4953;
  modelica_real tmp4954;
  modelica_real tmp4955;
  modelica_real tmp4956;
  modelica_real tmp4957;
  modelica_real tmp4958;
  modelica_real tmp4959;
  tmp4950 = (data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */);
  tmp4951 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4952 = (tmp4950 * tmp4950) + (tmp4951 * tmp4951);
  tmp4953 = 1.5;
  if(tmp4952 < 0.0 && tmp4953 != 0.0)
  {
    tmp4955 = modf(tmp4953, &tmp4956);
    
    if(tmp4955 > 0.5)
    {
      tmp4955 -= 1.0;
      tmp4956 += 1.0;
    }
    else if(tmp4955 < -0.5)
    {
      tmp4955 += 1.0;
      tmp4956 -= 1.0;
    }
    
    if(fabs(tmp4955) < 1e-10)
      tmp4954 = pow(tmp4952, tmp4956);
    else
    {
      tmp4958 = modf(1.0/tmp4953, &tmp4957);
      if(tmp4958 > 0.5)
      {
        tmp4958 -= 1.0;
        tmp4957 += 1.0;
      }
      else if(tmp4958 < -0.5)
      {
        tmp4958 += 1.0;
        tmp4957 -= 1.0;
      }
      if(fabs(tmp4958) < 1e-10 && ((unsigned long)tmp4957 & 1))
      {
        tmp4954 = -pow(-tmp4952, tmp4955)*pow(tmp4952, tmp4956);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4952, tmp4953);
      }
    }
  }
  else
  {
    tmp4954 = pow(tmp4952, tmp4953);
  }
  if(isnan(tmp4954) || isinf(tmp4954))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4952, tmp4953);
  }tmp4959 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4954,"(r_init[5] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4959 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[5] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4959);
    }
  }
  (data->simulationInfo->realParameter[509] /* omega_c[5] PARAM */) = sqrt(tmp4959);
  TRACE_POP
}

/*
equation index: 13993
type: SIMPLE_ASSIGN
r_init[4] = r_min + 4.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13993};
  (data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (4.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13994
type: SIMPLE_ASSIGN
omega_c[4] = sqrt(G * Md / (r_init[4] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13994(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13994};
  modelica_real tmp4960;
  modelica_real tmp4961;
  modelica_real tmp4962;
  modelica_real tmp4963;
  modelica_real tmp4964;
  modelica_real tmp4965;
  modelica_real tmp4966;
  modelica_real tmp4967;
  modelica_real tmp4968;
  modelica_real tmp4969;
  tmp4960 = (data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */);
  tmp4961 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4962 = (tmp4960 * tmp4960) + (tmp4961 * tmp4961);
  tmp4963 = 1.5;
  if(tmp4962 < 0.0 && tmp4963 != 0.0)
  {
    tmp4965 = modf(tmp4963, &tmp4966);
    
    if(tmp4965 > 0.5)
    {
      tmp4965 -= 1.0;
      tmp4966 += 1.0;
    }
    else if(tmp4965 < -0.5)
    {
      tmp4965 += 1.0;
      tmp4966 -= 1.0;
    }
    
    if(fabs(tmp4965) < 1e-10)
      tmp4964 = pow(tmp4962, tmp4966);
    else
    {
      tmp4968 = modf(1.0/tmp4963, &tmp4967);
      if(tmp4968 > 0.5)
      {
        tmp4968 -= 1.0;
        tmp4967 += 1.0;
      }
      else if(tmp4968 < -0.5)
      {
        tmp4968 += 1.0;
        tmp4967 -= 1.0;
      }
      if(fabs(tmp4968) < 1e-10 && ((unsigned long)tmp4967 & 1))
      {
        tmp4964 = -pow(-tmp4962, tmp4965)*pow(tmp4962, tmp4966);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4962, tmp4963);
      }
    }
  }
  else
  {
    tmp4964 = pow(tmp4962, tmp4963);
  }
  if(isnan(tmp4964) || isinf(tmp4964))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4962, tmp4963);
  }tmp4969 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4964,"(r_init[4] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4969 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[4] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4969);
    }
  }
  (data->simulationInfo->realParameter[508] /* omega_c[4] PARAM */) = sqrt(tmp4969);
  TRACE_POP
}

/*
equation index: 13995
type: SIMPLE_ASSIGN
r_init[3] = r_min + 3.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13995};
  (data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (3.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13996
type: SIMPLE_ASSIGN
omega_c[3] = sqrt(G * Md / (r_init[3] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13996};
  modelica_real tmp4970;
  modelica_real tmp4971;
  modelica_real tmp4972;
  modelica_real tmp4973;
  modelica_real tmp4974;
  modelica_real tmp4975;
  modelica_real tmp4976;
  modelica_real tmp4977;
  modelica_real tmp4978;
  modelica_real tmp4979;
  tmp4970 = (data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */);
  tmp4971 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4972 = (tmp4970 * tmp4970) + (tmp4971 * tmp4971);
  tmp4973 = 1.5;
  if(tmp4972 < 0.0 && tmp4973 != 0.0)
  {
    tmp4975 = modf(tmp4973, &tmp4976);
    
    if(tmp4975 > 0.5)
    {
      tmp4975 -= 1.0;
      tmp4976 += 1.0;
    }
    else if(tmp4975 < -0.5)
    {
      tmp4975 += 1.0;
      tmp4976 -= 1.0;
    }
    
    if(fabs(tmp4975) < 1e-10)
      tmp4974 = pow(tmp4972, tmp4976);
    else
    {
      tmp4978 = modf(1.0/tmp4973, &tmp4977);
      if(tmp4978 > 0.5)
      {
        tmp4978 -= 1.0;
        tmp4977 += 1.0;
      }
      else if(tmp4978 < -0.5)
      {
        tmp4978 += 1.0;
        tmp4977 -= 1.0;
      }
      if(fabs(tmp4978) < 1e-10 && ((unsigned long)tmp4977 & 1))
      {
        tmp4974 = -pow(-tmp4972, tmp4975)*pow(tmp4972, tmp4976);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4972, tmp4973);
      }
    }
  }
  else
  {
    tmp4974 = pow(tmp4972, tmp4973);
  }
  if(isnan(tmp4974) || isinf(tmp4974))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4972, tmp4973);
  }tmp4979 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4974,"(r_init[3] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4979 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[3] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4979);
    }
  }
  (data->simulationInfo->realParameter[507] /* omega_c[3] PARAM */) = sqrt(tmp4979);
  TRACE_POP
}

/*
equation index: 13997
type: SIMPLE_ASSIGN
r_init[2] = r_min + 2.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13997};
  (data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (2.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13998
type: SIMPLE_ASSIGN
omega_c[2] = sqrt(G * Md / (r_init[2] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13998(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13998};
  modelica_real tmp4980;
  modelica_real tmp4981;
  modelica_real tmp4982;
  modelica_real tmp4983;
  modelica_real tmp4984;
  modelica_real tmp4985;
  modelica_real tmp4986;
  modelica_real tmp4987;
  modelica_real tmp4988;
  modelica_real tmp4989;
  tmp4980 = (data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */);
  tmp4981 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4982 = (tmp4980 * tmp4980) + (tmp4981 * tmp4981);
  tmp4983 = 1.5;
  if(tmp4982 < 0.0 && tmp4983 != 0.0)
  {
    tmp4985 = modf(tmp4983, &tmp4986);
    
    if(tmp4985 > 0.5)
    {
      tmp4985 -= 1.0;
      tmp4986 += 1.0;
    }
    else if(tmp4985 < -0.5)
    {
      tmp4985 += 1.0;
      tmp4986 -= 1.0;
    }
    
    if(fabs(tmp4985) < 1e-10)
      tmp4984 = pow(tmp4982, tmp4986);
    else
    {
      tmp4988 = modf(1.0/tmp4983, &tmp4987);
      if(tmp4988 > 0.5)
      {
        tmp4988 -= 1.0;
        tmp4987 += 1.0;
      }
      else if(tmp4988 < -0.5)
      {
        tmp4988 += 1.0;
        tmp4987 -= 1.0;
      }
      if(fabs(tmp4988) < 1e-10 && ((unsigned long)tmp4987 & 1))
      {
        tmp4984 = -pow(-tmp4982, tmp4985)*pow(tmp4982, tmp4986);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4982, tmp4983);
      }
    }
  }
  else
  {
    tmp4984 = pow(tmp4982, tmp4983);
  }
  if(isnan(tmp4984) || isinf(tmp4984))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4982, tmp4983);
  }tmp4989 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4984,"(r_init[2] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4989 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[2] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4989);
    }
  }
  (data->simulationInfo->realParameter[506] /* omega_c[2] PARAM */) = sqrt(tmp4989);
  TRACE_POP
}

/*
equation index: 13999
type: SIMPLE_ASSIGN
r_init[1] = r_min + dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13999};
  (data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (data->simulationInfo->realParameter[504] /* dr PARAM */);
  TRACE_POP
}

/*
equation index: 14000
type: SIMPLE_ASSIGN
omega_c[1] = sqrt(G * Md / (r_init[1] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14000(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14000};
  modelica_real tmp4990;
  modelica_real tmp4991;
  modelica_real tmp4992;
  modelica_real tmp4993;
  modelica_real tmp4994;
  modelica_real tmp4995;
  modelica_real tmp4996;
  modelica_real tmp4997;
  modelica_real tmp4998;
  modelica_real tmp4999;
  tmp4990 = (data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */);
  tmp4991 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp4992 = (tmp4990 * tmp4990) + (tmp4991 * tmp4991);
  tmp4993 = 1.5;
  if(tmp4992 < 0.0 && tmp4993 != 0.0)
  {
    tmp4995 = modf(tmp4993, &tmp4996);
    
    if(tmp4995 > 0.5)
    {
      tmp4995 -= 1.0;
      tmp4996 += 1.0;
    }
    else if(tmp4995 < -0.5)
    {
      tmp4995 += 1.0;
      tmp4996 -= 1.0;
    }
    
    if(fabs(tmp4995) < 1e-10)
      tmp4994 = pow(tmp4992, tmp4996);
    else
    {
      tmp4998 = modf(1.0/tmp4993, &tmp4997);
      if(tmp4998 > 0.5)
      {
        tmp4998 -= 1.0;
        tmp4997 += 1.0;
      }
      else if(tmp4998 < -0.5)
      {
        tmp4998 += 1.0;
        tmp4997 -= 1.0;
      }
      if(fabs(tmp4998) < 1e-10 && ((unsigned long)tmp4997 & 1))
      {
        tmp4994 = -pow(-tmp4992, tmp4995)*pow(tmp4992, tmp4996);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4992, tmp4993);
      }
    }
  }
  else
  {
    tmp4994 = pow(tmp4992, tmp4993);
  }
  if(isnan(tmp4994) || isinf(tmp4994))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp4992, tmp4993);
  }tmp4999 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4994,"(r_init[1] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp4999 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[1] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp4999);
    }
  }
  (data->simulationInfo->realParameter[505] /* omega_c[1] PARAM */) = sqrt(tmp4999);
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_updateBoundParameters_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_13501(data, threadData);
  SpiralGalaxy_eqFunction_13502(data, threadData);
  SpiralGalaxy_eqFunction_13503(data, threadData);
  SpiralGalaxy_eqFunction_13504(data, threadData);
  SpiralGalaxy_eqFunction_13505(data, threadData);
  SpiralGalaxy_eqFunction_13506(data, threadData);
  SpiralGalaxy_eqFunction_13507(data, threadData);
  SpiralGalaxy_eqFunction_13508(data, threadData);
  SpiralGalaxy_eqFunction_13509(data, threadData);
  SpiralGalaxy_eqFunction_13510(data, threadData);
  SpiralGalaxy_eqFunction_13511(data, threadData);
  SpiralGalaxy_eqFunction_13512(data, threadData);
  SpiralGalaxy_eqFunction_13513(data, threadData);
  SpiralGalaxy_eqFunction_13514(data, threadData);
  SpiralGalaxy_eqFunction_13515(data, threadData);
  SpiralGalaxy_eqFunction_13516(data, threadData);
  SpiralGalaxy_eqFunction_13517(data, threadData);
  SpiralGalaxy_eqFunction_13518(data, threadData);
  SpiralGalaxy_eqFunction_13519(data, threadData);
  SpiralGalaxy_eqFunction_13520(data, threadData);
  SpiralGalaxy_eqFunction_13521(data, threadData);
  SpiralGalaxy_eqFunction_13522(data, threadData);
  SpiralGalaxy_eqFunction_13523(data, threadData);
  SpiralGalaxy_eqFunction_13524(data, threadData);
  SpiralGalaxy_eqFunction_13525(data, threadData);
  SpiralGalaxy_eqFunction_13526(data, threadData);
  SpiralGalaxy_eqFunction_13527(data, threadData);
  SpiralGalaxy_eqFunction_13528(data, threadData);
  SpiralGalaxy_eqFunction_13529(data, threadData);
  SpiralGalaxy_eqFunction_13530(data, threadData);
  SpiralGalaxy_eqFunction_13531(data, threadData);
  SpiralGalaxy_eqFunction_13532(data, threadData);
  SpiralGalaxy_eqFunction_13533(data, threadData);
  SpiralGalaxy_eqFunction_13534(data, threadData);
  SpiralGalaxy_eqFunction_13535(data, threadData);
  SpiralGalaxy_eqFunction_13536(data, threadData);
  SpiralGalaxy_eqFunction_13537(data, threadData);
  SpiralGalaxy_eqFunction_13538(data, threadData);
  SpiralGalaxy_eqFunction_13539(data, threadData);
  SpiralGalaxy_eqFunction_13540(data, threadData);
  SpiralGalaxy_eqFunction_13541(data, threadData);
  SpiralGalaxy_eqFunction_13542(data, threadData);
  SpiralGalaxy_eqFunction_13543(data, threadData);
  SpiralGalaxy_eqFunction_13544(data, threadData);
  SpiralGalaxy_eqFunction_13545(data, threadData);
  SpiralGalaxy_eqFunction_13546(data, threadData);
  SpiralGalaxy_eqFunction_13547(data, threadData);
  SpiralGalaxy_eqFunction_13548(data, threadData);
  SpiralGalaxy_eqFunction_13549(data, threadData);
  SpiralGalaxy_eqFunction_13550(data, threadData);
  SpiralGalaxy_eqFunction_13551(data, threadData);
  SpiralGalaxy_eqFunction_13552(data, threadData);
  SpiralGalaxy_eqFunction_13553(data, threadData);
  SpiralGalaxy_eqFunction_13554(data, threadData);
  SpiralGalaxy_eqFunction_13555(data, threadData);
  SpiralGalaxy_eqFunction_13556(data, threadData);
  SpiralGalaxy_eqFunction_13557(data, threadData);
  SpiralGalaxy_eqFunction_13558(data, threadData);
  SpiralGalaxy_eqFunction_13559(data, threadData);
  SpiralGalaxy_eqFunction_13560(data, threadData);
  SpiralGalaxy_eqFunction_13561(data, threadData);
  SpiralGalaxy_eqFunction_13562(data, threadData);
  SpiralGalaxy_eqFunction_13563(data, threadData);
  SpiralGalaxy_eqFunction_13564(data, threadData);
  SpiralGalaxy_eqFunction_13565(data, threadData);
  SpiralGalaxy_eqFunction_13566(data, threadData);
  SpiralGalaxy_eqFunction_13567(data, threadData);
  SpiralGalaxy_eqFunction_13568(data, threadData);
  SpiralGalaxy_eqFunction_13569(data, threadData);
  SpiralGalaxy_eqFunction_13570(data, threadData);
  SpiralGalaxy_eqFunction_13571(data, threadData);
  SpiralGalaxy_eqFunction_13572(data, threadData);
  SpiralGalaxy_eqFunction_13573(data, threadData);
  SpiralGalaxy_eqFunction_13574(data, threadData);
  SpiralGalaxy_eqFunction_13575(data, threadData);
  SpiralGalaxy_eqFunction_13576(data, threadData);
  SpiralGalaxy_eqFunction_13577(data, threadData);
  SpiralGalaxy_eqFunction_13578(data, threadData);
  SpiralGalaxy_eqFunction_13579(data, threadData);
  SpiralGalaxy_eqFunction_13580(data, threadData);
  SpiralGalaxy_eqFunction_13581(data, threadData);
  SpiralGalaxy_eqFunction_13582(data, threadData);
  SpiralGalaxy_eqFunction_13583(data, threadData);
  SpiralGalaxy_eqFunction_13584(data, threadData);
  SpiralGalaxy_eqFunction_13585(data, threadData);
  SpiralGalaxy_eqFunction_13586(data, threadData);
  SpiralGalaxy_eqFunction_13587(data, threadData);
  SpiralGalaxy_eqFunction_13588(data, threadData);
  SpiralGalaxy_eqFunction_13589(data, threadData);
  SpiralGalaxy_eqFunction_13590(data, threadData);
  SpiralGalaxy_eqFunction_13591(data, threadData);
  SpiralGalaxy_eqFunction_13592(data, threadData);
  SpiralGalaxy_eqFunction_13593(data, threadData);
  SpiralGalaxy_eqFunction_13594(data, threadData);
  SpiralGalaxy_eqFunction_13595(data, threadData);
  SpiralGalaxy_eqFunction_13596(data, threadData);
  SpiralGalaxy_eqFunction_13597(data, threadData);
  SpiralGalaxy_eqFunction_13598(data, threadData);
  SpiralGalaxy_eqFunction_13599(data, threadData);
  SpiralGalaxy_eqFunction_13600(data, threadData);
  SpiralGalaxy_eqFunction_13601(data, threadData);
  SpiralGalaxy_eqFunction_13602(data, threadData);
  SpiralGalaxy_eqFunction_13603(data, threadData);
  SpiralGalaxy_eqFunction_13604(data, threadData);
  SpiralGalaxy_eqFunction_13605(data, threadData);
  SpiralGalaxy_eqFunction_13606(data, threadData);
  SpiralGalaxy_eqFunction_13607(data, threadData);
  SpiralGalaxy_eqFunction_13608(data, threadData);
  SpiralGalaxy_eqFunction_13609(data, threadData);
  SpiralGalaxy_eqFunction_13610(data, threadData);
  SpiralGalaxy_eqFunction_13611(data, threadData);
  SpiralGalaxy_eqFunction_13612(data, threadData);
  SpiralGalaxy_eqFunction_13613(data, threadData);
  SpiralGalaxy_eqFunction_13614(data, threadData);
  SpiralGalaxy_eqFunction_13615(data, threadData);
  SpiralGalaxy_eqFunction_13616(data, threadData);
  SpiralGalaxy_eqFunction_13617(data, threadData);
  SpiralGalaxy_eqFunction_13618(data, threadData);
  SpiralGalaxy_eqFunction_13619(data, threadData);
  SpiralGalaxy_eqFunction_13620(data, threadData);
  SpiralGalaxy_eqFunction_13621(data, threadData);
  SpiralGalaxy_eqFunction_13622(data, threadData);
  SpiralGalaxy_eqFunction_13623(data, threadData);
  SpiralGalaxy_eqFunction_13624(data, threadData);
  SpiralGalaxy_eqFunction_13625(data, threadData);
  SpiralGalaxy_eqFunction_13626(data, threadData);
  SpiralGalaxy_eqFunction_13627(data, threadData);
  SpiralGalaxy_eqFunction_13628(data, threadData);
  SpiralGalaxy_eqFunction_13629(data, threadData);
  SpiralGalaxy_eqFunction_13630(data, threadData);
  SpiralGalaxy_eqFunction_13631(data, threadData);
  SpiralGalaxy_eqFunction_13632(data, threadData);
  SpiralGalaxy_eqFunction_13633(data, threadData);
  SpiralGalaxy_eqFunction_13634(data, threadData);
  SpiralGalaxy_eqFunction_13635(data, threadData);
  SpiralGalaxy_eqFunction_13636(data, threadData);
  SpiralGalaxy_eqFunction_13637(data, threadData);
  SpiralGalaxy_eqFunction_13638(data, threadData);
  SpiralGalaxy_eqFunction_13639(data, threadData);
  SpiralGalaxy_eqFunction_13640(data, threadData);
  SpiralGalaxy_eqFunction_13641(data, threadData);
  SpiralGalaxy_eqFunction_13642(data, threadData);
  SpiralGalaxy_eqFunction_13643(data, threadData);
  SpiralGalaxy_eqFunction_13644(data, threadData);
  SpiralGalaxy_eqFunction_13645(data, threadData);
  SpiralGalaxy_eqFunction_13646(data, threadData);
  SpiralGalaxy_eqFunction_13647(data, threadData);
  SpiralGalaxy_eqFunction_13648(data, threadData);
  SpiralGalaxy_eqFunction_13649(data, threadData);
  SpiralGalaxy_eqFunction_13650(data, threadData);
  SpiralGalaxy_eqFunction_13651(data, threadData);
  SpiralGalaxy_eqFunction_13652(data, threadData);
  SpiralGalaxy_eqFunction_13653(data, threadData);
  SpiralGalaxy_eqFunction_13654(data, threadData);
  SpiralGalaxy_eqFunction_13655(data, threadData);
  SpiralGalaxy_eqFunction_13656(data, threadData);
  SpiralGalaxy_eqFunction_13657(data, threadData);
  SpiralGalaxy_eqFunction_13658(data, threadData);
  SpiralGalaxy_eqFunction_13659(data, threadData);
  SpiralGalaxy_eqFunction_13660(data, threadData);
  SpiralGalaxy_eqFunction_13661(data, threadData);
  SpiralGalaxy_eqFunction_13662(data, threadData);
  SpiralGalaxy_eqFunction_13663(data, threadData);
  SpiralGalaxy_eqFunction_13664(data, threadData);
  SpiralGalaxy_eqFunction_13665(data, threadData);
  SpiralGalaxy_eqFunction_13666(data, threadData);
  SpiralGalaxy_eqFunction_13667(data, threadData);
  SpiralGalaxy_eqFunction_13668(data, threadData);
  SpiralGalaxy_eqFunction_13669(data, threadData);
  SpiralGalaxy_eqFunction_13670(data, threadData);
  SpiralGalaxy_eqFunction_13671(data, threadData);
  SpiralGalaxy_eqFunction_13672(data, threadData);
  SpiralGalaxy_eqFunction_13673(data, threadData);
  SpiralGalaxy_eqFunction_13674(data, threadData);
  SpiralGalaxy_eqFunction_13675(data, threadData);
  SpiralGalaxy_eqFunction_13676(data, threadData);
  SpiralGalaxy_eqFunction_13677(data, threadData);
  SpiralGalaxy_eqFunction_13678(data, threadData);
  SpiralGalaxy_eqFunction_13679(data, threadData);
  SpiralGalaxy_eqFunction_13680(data, threadData);
  SpiralGalaxy_eqFunction_13681(data, threadData);
  SpiralGalaxy_eqFunction_13682(data, threadData);
  SpiralGalaxy_eqFunction_13683(data, threadData);
  SpiralGalaxy_eqFunction_13684(data, threadData);
  SpiralGalaxy_eqFunction_13685(data, threadData);
  SpiralGalaxy_eqFunction_13686(data, threadData);
  SpiralGalaxy_eqFunction_13687(data, threadData);
  SpiralGalaxy_eqFunction_13688(data, threadData);
  SpiralGalaxy_eqFunction_13689(data, threadData);
  SpiralGalaxy_eqFunction_13690(data, threadData);
  SpiralGalaxy_eqFunction_13691(data, threadData);
  SpiralGalaxy_eqFunction_13692(data, threadData);
  SpiralGalaxy_eqFunction_13693(data, threadData);
  SpiralGalaxy_eqFunction_13694(data, threadData);
  SpiralGalaxy_eqFunction_13695(data, threadData);
  SpiralGalaxy_eqFunction_13696(data, threadData);
  SpiralGalaxy_eqFunction_13697(data, threadData);
  SpiralGalaxy_eqFunction_13698(data, threadData);
  SpiralGalaxy_eqFunction_13699(data, threadData);
  SpiralGalaxy_eqFunction_13700(data, threadData);
  SpiralGalaxy_eqFunction_13701(data, threadData);
  SpiralGalaxy_eqFunction_13702(data, threadData);
  SpiralGalaxy_eqFunction_13703(data, threadData);
  SpiralGalaxy_eqFunction_13704(data, threadData);
  SpiralGalaxy_eqFunction_13705(data, threadData);
  SpiralGalaxy_eqFunction_13706(data, threadData);
  SpiralGalaxy_eqFunction_13707(data, threadData);
  SpiralGalaxy_eqFunction_13708(data, threadData);
  SpiralGalaxy_eqFunction_13709(data, threadData);
  SpiralGalaxy_eqFunction_13710(data, threadData);
  SpiralGalaxy_eqFunction_13711(data, threadData);
  SpiralGalaxy_eqFunction_13712(data, threadData);
  SpiralGalaxy_eqFunction_13713(data, threadData);
  SpiralGalaxy_eqFunction_13714(data, threadData);
  SpiralGalaxy_eqFunction_13715(data, threadData);
  SpiralGalaxy_eqFunction_13716(data, threadData);
  SpiralGalaxy_eqFunction_13717(data, threadData);
  SpiralGalaxy_eqFunction_13718(data, threadData);
  SpiralGalaxy_eqFunction_13719(data, threadData);
  SpiralGalaxy_eqFunction_13720(data, threadData);
  SpiralGalaxy_eqFunction_13721(data, threadData);
  SpiralGalaxy_eqFunction_13722(data, threadData);
  SpiralGalaxy_eqFunction_13723(data, threadData);
  SpiralGalaxy_eqFunction_13724(data, threadData);
  SpiralGalaxy_eqFunction_13725(data, threadData);
  SpiralGalaxy_eqFunction_13726(data, threadData);
  SpiralGalaxy_eqFunction_13727(data, threadData);
  SpiralGalaxy_eqFunction_13728(data, threadData);
  SpiralGalaxy_eqFunction_13729(data, threadData);
  SpiralGalaxy_eqFunction_13730(data, threadData);
  SpiralGalaxy_eqFunction_13731(data, threadData);
  SpiralGalaxy_eqFunction_13732(data, threadData);
  SpiralGalaxy_eqFunction_13733(data, threadData);
  SpiralGalaxy_eqFunction_13734(data, threadData);
  SpiralGalaxy_eqFunction_13735(data, threadData);
  SpiralGalaxy_eqFunction_13736(data, threadData);
  SpiralGalaxy_eqFunction_13737(data, threadData);
  SpiralGalaxy_eqFunction_13738(data, threadData);
  SpiralGalaxy_eqFunction_13739(data, threadData);
  SpiralGalaxy_eqFunction_13740(data, threadData);
  SpiralGalaxy_eqFunction_13741(data, threadData);
  SpiralGalaxy_eqFunction_13742(data, threadData);
  SpiralGalaxy_eqFunction_13743(data, threadData);
  SpiralGalaxy_eqFunction_13744(data, threadData);
  SpiralGalaxy_eqFunction_13745(data, threadData);
  SpiralGalaxy_eqFunction_13746(data, threadData);
  SpiralGalaxy_eqFunction_13747(data, threadData);
  SpiralGalaxy_eqFunction_13748(data, threadData);
  SpiralGalaxy_eqFunction_13749(data, threadData);
  SpiralGalaxy_eqFunction_13750(data, threadData);
  SpiralGalaxy_eqFunction_13751(data, threadData);
  SpiralGalaxy_eqFunction_13752(data, threadData);
  SpiralGalaxy_eqFunction_13753(data, threadData);
  SpiralGalaxy_eqFunction_13754(data, threadData);
  SpiralGalaxy_eqFunction_13755(data, threadData);
  SpiralGalaxy_eqFunction_13756(data, threadData);
  SpiralGalaxy_eqFunction_13757(data, threadData);
  SpiralGalaxy_eqFunction_13758(data, threadData);
  SpiralGalaxy_eqFunction_13759(data, threadData);
  SpiralGalaxy_eqFunction_13760(data, threadData);
  SpiralGalaxy_eqFunction_13761(data, threadData);
  SpiralGalaxy_eqFunction_13762(data, threadData);
  SpiralGalaxy_eqFunction_13763(data, threadData);
  SpiralGalaxy_eqFunction_13764(data, threadData);
  SpiralGalaxy_eqFunction_13765(data, threadData);
  SpiralGalaxy_eqFunction_13766(data, threadData);
  SpiralGalaxy_eqFunction_13767(data, threadData);
  SpiralGalaxy_eqFunction_13768(data, threadData);
  SpiralGalaxy_eqFunction_13769(data, threadData);
  SpiralGalaxy_eqFunction_13770(data, threadData);
  SpiralGalaxy_eqFunction_13771(data, threadData);
  SpiralGalaxy_eqFunction_13772(data, threadData);
  SpiralGalaxy_eqFunction_13773(data, threadData);
  SpiralGalaxy_eqFunction_13774(data, threadData);
  SpiralGalaxy_eqFunction_13775(data, threadData);
  SpiralGalaxy_eqFunction_13776(data, threadData);
  SpiralGalaxy_eqFunction_13777(data, threadData);
  SpiralGalaxy_eqFunction_13778(data, threadData);
  SpiralGalaxy_eqFunction_13779(data, threadData);
  SpiralGalaxy_eqFunction_13780(data, threadData);
  SpiralGalaxy_eqFunction_13781(data, threadData);
  SpiralGalaxy_eqFunction_13782(data, threadData);
  SpiralGalaxy_eqFunction_13783(data, threadData);
  SpiralGalaxy_eqFunction_13784(data, threadData);
  SpiralGalaxy_eqFunction_13785(data, threadData);
  SpiralGalaxy_eqFunction_13786(data, threadData);
  SpiralGalaxy_eqFunction_13787(data, threadData);
  SpiralGalaxy_eqFunction_13788(data, threadData);
  SpiralGalaxy_eqFunction_13789(data, threadData);
  SpiralGalaxy_eqFunction_13790(data, threadData);
  SpiralGalaxy_eqFunction_13791(data, threadData);
  SpiralGalaxy_eqFunction_13792(data, threadData);
  SpiralGalaxy_eqFunction_13793(data, threadData);
  SpiralGalaxy_eqFunction_13794(data, threadData);
  SpiralGalaxy_eqFunction_13795(data, threadData);
  SpiralGalaxy_eqFunction_13796(data, threadData);
  SpiralGalaxy_eqFunction_13797(data, threadData);
  SpiralGalaxy_eqFunction_13798(data, threadData);
  SpiralGalaxy_eqFunction_13799(data, threadData);
  SpiralGalaxy_eqFunction_13800(data, threadData);
  SpiralGalaxy_eqFunction_13801(data, threadData);
  SpiralGalaxy_eqFunction_13802(data, threadData);
  SpiralGalaxy_eqFunction_13803(data, threadData);
  SpiralGalaxy_eqFunction_13804(data, threadData);
  SpiralGalaxy_eqFunction_13805(data, threadData);
  SpiralGalaxy_eqFunction_13806(data, threadData);
  SpiralGalaxy_eqFunction_13807(data, threadData);
  SpiralGalaxy_eqFunction_13808(data, threadData);
  SpiralGalaxy_eqFunction_13809(data, threadData);
  SpiralGalaxy_eqFunction_13810(data, threadData);
  SpiralGalaxy_eqFunction_13811(data, threadData);
  SpiralGalaxy_eqFunction_13812(data, threadData);
  SpiralGalaxy_eqFunction_13813(data, threadData);
  SpiralGalaxy_eqFunction_13814(data, threadData);
  SpiralGalaxy_eqFunction_13815(data, threadData);
  SpiralGalaxy_eqFunction_13816(data, threadData);
  SpiralGalaxy_eqFunction_13817(data, threadData);
  SpiralGalaxy_eqFunction_13818(data, threadData);
  SpiralGalaxy_eqFunction_13819(data, threadData);
  SpiralGalaxy_eqFunction_13820(data, threadData);
  SpiralGalaxy_eqFunction_13821(data, threadData);
  SpiralGalaxy_eqFunction_13822(data, threadData);
  SpiralGalaxy_eqFunction_13823(data, threadData);
  SpiralGalaxy_eqFunction_13824(data, threadData);
  SpiralGalaxy_eqFunction_13825(data, threadData);
  SpiralGalaxy_eqFunction_13826(data, threadData);
  SpiralGalaxy_eqFunction_13827(data, threadData);
  SpiralGalaxy_eqFunction_13828(data, threadData);
  SpiralGalaxy_eqFunction_13829(data, threadData);
  SpiralGalaxy_eqFunction_13830(data, threadData);
  SpiralGalaxy_eqFunction_13831(data, threadData);
  SpiralGalaxy_eqFunction_13832(data, threadData);
  SpiralGalaxy_eqFunction_13833(data, threadData);
  SpiralGalaxy_eqFunction_13834(data, threadData);
  SpiralGalaxy_eqFunction_13835(data, threadData);
  SpiralGalaxy_eqFunction_13836(data, threadData);
  SpiralGalaxy_eqFunction_13837(data, threadData);
  SpiralGalaxy_eqFunction_13838(data, threadData);
  SpiralGalaxy_eqFunction_13839(data, threadData);
  SpiralGalaxy_eqFunction_13840(data, threadData);
  SpiralGalaxy_eqFunction_13841(data, threadData);
  SpiralGalaxy_eqFunction_13842(data, threadData);
  SpiralGalaxy_eqFunction_13843(data, threadData);
  SpiralGalaxy_eqFunction_13844(data, threadData);
  SpiralGalaxy_eqFunction_13845(data, threadData);
  SpiralGalaxy_eqFunction_13846(data, threadData);
  SpiralGalaxy_eqFunction_13847(data, threadData);
  SpiralGalaxy_eqFunction_13848(data, threadData);
  SpiralGalaxy_eqFunction_13849(data, threadData);
  SpiralGalaxy_eqFunction_13850(data, threadData);
  SpiralGalaxy_eqFunction_13851(data, threadData);
  SpiralGalaxy_eqFunction_13852(data, threadData);
  SpiralGalaxy_eqFunction_13853(data, threadData);
  SpiralGalaxy_eqFunction_13854(data, threadData);
  SpiralGalaxy_eqFunction_13855(data, threadData);
  SpiralGalaxy_eqFunction_13856(data, threadData);
  SpiralGalaxy_eqFunction_13857(data, threadData);
  SpiralGalaxy_eqFunction_13858(data, threadData);
  SpiralGalaxy_eqFunction_13859(data, threadData);
  SpiralGalaxy_eqFunction_13860(data, threadData);
  SpiralGalaxy_eqFunction_13861(data, threadData);
  SpiralGalaxy_eqFunction_13862(data, threadData);
  SpiralGalaxy_eqFunction_13863(data, threadData);
  SpiralGalaxy_eqFunction_13864(data, threadData);
  SpiralGalaxy_eqFunction_13865(data, threadData);
  SpiralGalaxy_eqFunction_13866(data, threadData);
  SpiralGalaxy_eqFunction_13867(data, threadData);
  SpiralGalaxy_eqFunction_13868(data, threadData);
  SpiralGalaxy_eqFunction_13869(data, threadData);
  SpiralGalaxy_eqFunction_13870(data, threadData);
  SpiralGalaxy_eqFunction_13871(data, threadData);
  SpiralGalaxy_eqFunction_13872(data, threadData);
  SpiralGalaxy_eqFunction_13873(data, threadData);
  SpiralGalaxy_eqFunction_13874(data, threadData);
  SpiralGalaxy_eqFunction_13875(data, threadData);
  SpiralGalaxy_eqFunction_13876(data, threadData);
  SpiralGalaxy_eqFunction_13877(data, threadData);
  SpiralGalaxy_eqFunction_13878(data, threadData);
  SpiralGalaxy_eqFunction_13879(data, threadData);
  SpiralGalaxy_eqFunction_13880(data, threadData);
  SpiralGalaxy_eqFunction_13881(data, threadData);
  SpiralGalaxy_eqFunction_13882(data, threadData);
  SpiralGalaxy_eqFunction_13883(data, threadData);
  SpiralGalaxy_eqFunction_13884(data, threadData);
  SpiralGalaxy_eqFunction_13885(data, threadData);
  SpiralGalaxy_eqFunction_13886(data, threadData);
  SpiralGalaxy_eqFunction_13887(data, threadData);
  SpiralGalaxy_eqFunction_13888(data, threadData);
  SpiralGalaxy_eqFunction_13889(data, threadData);
  SpiralGalaxy_eqFunction_13890(data, threadData);
  SpiralGalaxy_eqFunction_13891(data, threadData);
  SpiralGalaxy_eqFunction_13892(data, threadData);
  SpiralGalaxy_eqFunction_13893(data, threadData);
  SpiralGalaxy_eqFunction_13894(data, threadData);
  SpiralGalaxy_eqFunction_13895(data, threadData);
  SpiralGalaxy_eqFunction_13896(data, threadData);
  SpiralGalaxy_eqFunction_13897(data, threadData);
  SpiralGalaxy_eqFunction_13898(data, threadData);
  SpiralGalaxy_eqFunction_13899(data, threadData);
  SpiralGalaxy_eqFunction_13900(data, threadData);
  SpiralGalaxy_eqFunction_13901(data, threadData);
  SpiralGalaxy_eqFunction_13902(data, threadData);
  SpiralGalaxy_eqFunction_13903(data, threadData);
  SpiralGalaxy_eqFunction_13904(data, threadData);
  SpiralGalaxy_eqFunction_13905(data, threadData);
  SpiralGalaxy_eqFunction_13906(data, threadData);
  SpiralGalaxy_eqFunction_13907(data, threadData);
  SpiralGalaxy_eqFunction_13908(data, threadData);
  SpiralGalaxy_eqFunction_13909(data, threadData);
  SpiralGalaxy_eqFunction_13910(data, threadData);
  SpiralGalaxy_eqFunction_13911(data, threadData);
  SpiralGalaxy_eqFunction_13912(data, threadData);
  SpiralGalaxy_eqFunction_13913(data, threadData);
  SpiralGalaxy_eqFunction_13914(data, threadData);
  SpiralGalaxy_eqFunction_13915(data, threadData);
  SpiralGalaxy_eqFunction_13916(data, threadData);
  SpiralGalaxy_eqFunction_13917(data, threadData);
  SpiralGalaxy_eqFunction_13918(data, threadData);
  SpiralGalaxy_eqFunction_13919(data, threadData);
  SpiralGalaxy_eqFunction_13920(data, threadData);
  SpiralGalaxy_eqFunction_13921(data, threadData);
  SpiralGalaxy_eqFunction_13922(data, threadData);
  SpiralGalaxy_eqFunction_13923(data, threadData);
  SpiralGalaxy_eqFunction_13924(data, threadData);
  SpiralGalaxy_eqFunction_13925(data, threadData);
  SpiralGalaxy_eqFunction_13926(data, threadData);
  SpiralGalaxy_eqFunction_13927(data, threadData);
  SpiralGalaxy_eqFunction_13928(data, threadData);
  SpiralGalaxy_eqFunction_13929(data, threadData);
  SpiralGalaxy_eqFunction_13930(data, threadData);
  SpiralGalaxy_eqFunction_13931(data, threadData);
  SpiralGalaxy_eqFunction_13932(data, threadData);
  SpiralGalaxy_eqFunction_13933(data, threadData);
  SpiralGalaxy_eqFunction_13934(data, threadData);
  SpiralGalaxy_eqFunction_13935(data, threadData);
  SpiralGalaxy_eqFunction_13936(data, threadData);
  SpiralGalaxy_eqFunction_13937(data, threadData);
  SpiralGalaxy_eqFunction_13938(data, threadData);
  SpiralGalaxy_eqFunction_13939(data, threadData);
  SpiralGalaxy_eqFunction_13940(data, threadData);
  SpiralGalaxy_eqFunction_13941(data, threadData);
  SpiralGalaxy_eqFunction_13942(data, threadData);
  SpiralGalaxy_eqFunction_13943(data, threadData);
  SpiralGalaxy_eqFunction_13944(data, threadData);
  SpiralGalaxy_eqFunction_13945(data, threadData);
  SpiralGalaxy_eqFunction_13946(data, threadData);
  SpiralGalaxy_eqFunction_13947(data, threadData);
  SpiralGalaxy_eqFunction_13948(data, threadData);
  SpiralGalaxy_eqFunction_13949(data, threadData);
  SpiralGalaxy_eqFunction_13950(data, threadData);
  SpiralGalaxy_eqFunction_13951(data, threadData);
  SpiralGalaxy_eqFunction_13952(data, threadData);
  SpiralGalaxy_eqFunction_13953(data, threadData);
  SpiralGalaxy_eqFunction_13954(data, threadData);
  SpiralGalaxy_eqFunction_13955(data, threadData);
  SpiralGalaxy_eqFunction_13956(data, threadData);
  SpiralGalaxy_eqFunction_13957(data, threadData);
  SpiralGalaxy_eqFunction_13958(data, threadData);
  SpiralGalaxy_eqFunction_13959(data, threadData);
  SpiralGalaxy_eqFunction_13960(data, threadData);
  SpiralGalaxy_eqFunction_13961(data, threadData);
  SpiralGalaxy_eqFunction_13962(data, threadData);
  SpiralGalaxy_eqFunction_13963(data, threadData);
  SpiralGalaxy_eqFunction_13964(data, threadData);
  SpiralGalaxy_eqFunction_13965(data, threadData);
  SpiralGalaxy_eqFunction_13966(data, threadData);
  SpiralGalaxy_eqFunction_13967(data, threadData);
  SpiralGalaxy_eqFunction_13968(data, threadData);
  SpiralGalaxy_eqFunction_13969(data, threadData);
  SpiralGalaxy_eqFunction_13970(data, threadData);
  SpiralGalaxy_eqFunction_13971(data, threadData);
  SpiralGalaxy_eqFunction_13972(data, threadData);
  SpiralGalaxy_eqFunction_13973(data, threadData);
  SpiralGalaxy_eqFunction_13974(data, threadData);
  SpiralGalaxy_eqFunction_13975(data, threadData);
  SpiralGalaxy_eqFunction_13976(data, threadData);
  SpiralGalaxy_eqFunction_13977(data, threadData);
  SpiralGalaxy_eqFunction_13978(data, threadData);
  SpiralGalaxy_eqFunction_13979(data, threadData);
  SpiralGalaxy_eqFunction_13980(data, threadData);
  SpiralGalaxy_eqFunction_13981(data, threadData);
  SpiralGalaxy_eqFunction_13982(data, threadData);
  SpiralGalaxy_eqFunction_13983(data, threadData);
  SpiralGalaxy_eqFunction_13984(data, threadData);
  SpiralGalaxy_eqFunction_13985(data, threadData);
  SpiralGalaxy_eqFunction_13986(data, threadData);
  SpiralGalaxy_eqFunction_13987(data, threadData);
  SpiralGalaxy_eqFunction_13988(data, threadData);
  SpiralGalaxy_eqFunction_13989(data, threadData);
  SpiralGalaxy_eqFunction_13990(data, threadData);
  SpiralGalaxy_eqFunction_13991(data, threadData);
  SpiralGalaxy_eqFunction_13992(data, threadData);
  SpiralGalaxy_eqFunction_13993(data, threadData);
  SpiralGalaxy_eqFunction_13994(data, threadData);
  SpiralGalaxy_eqFunction_13995(data, threadData);
  SpiralGalaxy_eqFunction_13996(data, threadData);
  SpiralGalaxy_eqFunction_13997(data, threadData);
  SpiralGalaxy_eqFunction_13998(data, threadData);
  SpiralGalaxy_eqFunction_13999(data, threadData);
  SpiralGalaxy_eqFunction_14000(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif