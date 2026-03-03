#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 13001
type: SIMPLE_ASSIGN
r_init[500] = r_min + 500.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13001(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13001};
  (data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (500.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13002
type: SIMPLE_ASSIGN
omega_c[500] = sqrt(G * Md / (r_init[500] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13002(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13002};
  modelica_real tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  tmp0 = (data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */);
  tmp1 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2 = (tmp0 * tmp0) + (tmp1 * tmp1);
  tmp3 = 1.5;
  if(tmp2 < 0.0 && tmp3 != 0.0)
  {
    tmp5 = modf(tmp3, &tmp6);
    
    if(tmp5 > 0.5)
    {
      tmp5 -= 1.0;
      tmp6 += 1.0;
    }
    else if(tmp5 < -0.5)
    {
      tmp5 += 1.0;
      tmp6 -= 1.0;
    }
    
    if(fabs(tmp5) < 1e-10)
      tmp4 = pow(tmp2, tmp6);
    else
    {
      tmp8 = modf(1.0/tmp3, &tmp7);
      if(tmp8 > 0.5)
      {
        tmp8 -= 1.0;
        tmp7 += 1.0;
      }
      else if(tmp8 < -0.5)
      {
        tmp8 += 1.0;
        tmp7 -= 1.0;
      }
      if(fabs(tmp8) < 1e-10 && ((unsigned long)tmp7 & 1))
      {
        tmp4 = -pow(-tmp2, tmp5)*pow(tmp2, tmp6);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2, tmp3);
      }
    }
  }
  else
  {
    tmp4 = pow(tmp2, tmp3);
  }
  if(isnan(tmp4) || isinf(tmp4))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2, tmp3);
  }tmp9 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp4,"(r_init[500] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp9 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[500] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp9);
    }
  }
  (data->simulationInfo->realParameter[1004] /* omega_c[500] PARAM */) = sqrt(tmp9);
  TRACE_POP
}

/*
equation index: 13003
type: SIMPLE_ASSIGN
r_init[499] = r_min + 499.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13003};
  (data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (499.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13004
type: SIMPLE_ASSIGN
omega_c[499] = sqrt(G * Md / (r_init[499] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13004(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13004};
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  tmp10 = (data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */);
  tmp11 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp12 = (tmp10 * tmp10) + (tmp11 * tmp11);
  tmp13 = 1.5;
  if(tmp12 < 0.0 && tmp13 != 0.0)
  {
    tmp15 = modf(tmp13, &tmp16);
    
    if(tmp15 > 0.5)
    {
      tmp15 -= 1.0;
      tmp16 += 1.0;
    }
    else if(tmp15 < -0.5)
    {
      tmp15 += 1.0;
      tmp16 -= 1.0;
    }
    
    if(fabs(tmp15) < 1e-10)
      tmp14 = pow(tmp12, tmp16);
    else
    {
      tmp18 = modf(1.0/tmp13, &tmp17);
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp17 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp17 -= 1.0;
      }
      if(fabs(tmp18) < 1e-10 && ((unsigned long)tmp17 & 1))
      {
        tmp14 = -pow(-tmp12, tmp15)*pow(tmp12, tmp16);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp12, tmp13);
      }
    }
  }
  else
  {
    tmp14 = pow(tmp12, tmp13);
  }
  if(isnan(tmp14) || isinf(tmp14))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp12, tmp13);
  }tmp19 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp14,"(r_init[499] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp19 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[499] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp19);
    }
  }
  (data->simulationInfo->realParameter[1003] /* omega_c[499] PARAM */) = sqrt(tmp19);
  TRACE_POP
}

/*
equation index: 13005
type: SIMPLE_ASSIGN
r_init[498] = r_min + 498.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13005};
  (data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (498.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13006
type: SIMPLE_ASSIGN
omega_c[498] = sqrt(G * Md / (r_init[498] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13006(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13006};
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_real tmp22;
  modelica_real tmp23;
  modelica_real tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  tmp20 = (data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */);
  tmp21 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp22 = (tmp20 * tmp20) + (tmp21 * tmp21);
  tmp23 = 1.5;
  if(tmp22 < 0.0 && tmp23 != 0.0)
  {
    tmp25 = modf(tmp23, &tmp26);
    
    if(tmp25 > 0.5)
    {
      tmp25 -= 1.0;
      tmp26 += 1.0;
    }
    else if(tmp25 < -0.5)
    {
      tmp25 += 1.0;
      tmp26 -= 1.0;
    }
    
    if(fabs(tmp25) < 1e-10)
      tmp24 = pow(tmp22, tmp26);
    else
    {
      tmp28 = modf(1.0/tmp23, &tmp27);
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp27 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp27 -= 1.0;
      }
      if(fabs(tmp28) < 1e-10 && ((unsigned long)tmp27 & 1))
      {
        tmp24 = -pow(-tmp22, tmp25)*pow(tmp22, tmp26);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp22, tmp23);
      }
    }
  }
  else
  {
    tmp24 = pow(tmp22, tmp23);
  }
  if(isnan(tmp24) || isinf(tmp24))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp22, tmp23);
  }tmp29 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp24,"(r_init[498] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp29 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[498] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp29);
    }
  }
  (data->simulationInfo->realParameter[1002] /* omega_c[498] PARAM */) = sqrt(tmp29);
  TRACE_POP
}

/*
equation index: 13007
type: SIMPLE_ASSIGN
r_init[497] = r_min + 497.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13007};
  (data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (497.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13008
type: SIMPLE_ASSIGN
omega_c[497] = sqrt(G * Md / (r_init[497] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13008(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13008};
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  modelica_real tmp33;
  modelica_real tmp34;
  modelica_real tmp35;
  modelica_real tmp36;
  modelica_real tmp37;
  modelica_real tmp38;
  modelica_real tmp39;
  tmp30 = (data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */);
  tmp31 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp32 = (tmp30 * tmp30) + (tmp31 * tmp31);
  tmp33 = 1.5;
  if(tmp32 < 0.0 && tmp33 != 0.0)
  {
    tmp35 = modf(tmp33, &tmp36);
    
    if(tmp35 > 0.5)
    {
      tmp35 -= 1.0;
      tmp36 += 1.0;
    }
    else if(tmp35 < -0.5)
    {
      tmp35 += 1.0;
      tmp36 -= 1.0;
    }
    
    if(fabs(tmp35) < 1e-10)
      tmp34 = pow(tmp32, tmp36);
    else
    {
      tmp38 = modf(1.0/tmp33, &tmp37);
      if(tmp38 > 0.5)
      {
        tmp38 -= 1.0;
        tmp37 += 1.0;
      }
      else if(tmp38 < -0.5)
      {
        tmp38 += 1.0;
        tmp37 -= 1.0;
      }
      if(fabs(tmp38) < 1e-10 && ((unsigned long)tmp37 & 1))
      {
        tmp34 = -pow(-tmp32, tmp35)*pow(tmp32, tmp36);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp32, tmp33);
      }
    }
  }
  else
  {
    tmp34 = pow(tmp32, tmp33);
  }
  if(isnan(tmp34) || isinf(tmp34))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp32, tmp33);
  }tmp39 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp34,"(r_init[497] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp39 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[497] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp39);
    }
  }
  (data->simulationInfo->realParameter[1001] /* omega_c[497] PARAM */) = sqrt(tmp39);
  TRACE_POP
}

/*
equation index: 13009
type: SIMPLE_ASSIGN
r_init[496] = r_min + 496.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13009};
  (data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (496.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13010
type: SIMPLE_ASSIGN
omega_c[496] = sqrt(G * Md / (r_init[496] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13010(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13010};
  modelica_real tmp40;
  modelica_real tmp41;
  modelica_real tmp42;
  modelica_real tmp43;
  modelica_real tmp44;
  modelica_real tmp45;
  modelica_real tmp46;
  modelica_real tmp47;
  modelica_real tmp48;
  modelica_real tmp49;
  tmp40 = (data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */);
  tmp41 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp42 = (tmp40 * tmp40) + (tmp41 * tmp41);
  tmp43 = 1.5;
  if(tmp42 < 0.0 && tmp43 != 0.0)
  {
    tmp45 = modf(tmp43, &tmp46);
    
    if(tmp45 > 0.5)
    {
      tmp45 -= 1.0;
      tmp46 += 1.0;
    }
    else if(tmp45 < -0.5)
    {
      tmp45 += 1.0;
      tmp46 -= 1.0;
    }
    
    if(fabs(tmp45) < 1e-10)
      tmp44 = pow(tmp42, tmp46);
    else
    {
      tmp48 = modf(1.0/tmp43, &tmp47);
      if(tmp48 > 0.5)
      {
        tmp48 -= 1.0;
        tmp47 += 1.0;
      }
      else if(tmp48 < -0.5)
      {
        tmp48 += 1.0;
        tmp47 -= 1.0;
      }
      if(fabs(tmp48) < 1e-10 && ((unsigned long)tmp47 & 1))
      {
        tmp44 = -pow(-tmp42, tmp45)*pow(tmp42, tmp46);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp42, tmp43);
      }
    }
  }
  else
  {
    tmp44 = pow(tmp42, tmp43);
  }
  if(isnan(tmp44) || isinf(tmp44))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp42, tmp43);
  }tmp49 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp44,"(r_init[496] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp49 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[496] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp49);
    }
  }
  (data->simulationInfo->realParameter[1000] /* omega_c[496] PARAM */) = sqrt(tmp49);
  TRACE_POP
}

/*
equation index: 13011
type: SIMPLE_ASSIGN
r_init[495] = r_min + 495.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13011(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13011};
  (data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (495.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13012
type: SIMPLE_ASSIGN
omega_c[495] = sqrt(G * Md / (r_init[495] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13012};
  modelica_real tmp50;
  modelica_real tmp51;
  modelica_real tmp52;
  modelica_real tmp53;
  modelica_real tmp54;
  modelica_real tmp55;
  modelica_real tmp56;
  modelica_real tmp57;
  modelica_real tmp58;
  modelica_real tmp59;
  tmp50 = (data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */);
  tmp51 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp52 = (tmp50 * tmp50) + (tmp51 * tmp51);
  tmp53 = 1.5;
  if(tmp52 < 0.0 && tmp53 != 0.0)
  {
    tmp55 = modf(tmp53, &tmp56);
    
    if(tmp55 > 0.5)
    {
      tmp55 -= 1.0;
      tmp56 += 1.0;
    }
    else if(tmp55 < -0.5)
    {
      tmp55 += 1.0;
      tmp56 -= 1.0;
    }
    
    if(fabs(tmp55) < 1e-10)
      tmp54 = pow(tmp52, tmp56);
    else
    {
      tmp58 = modf(1.0/tmp53, &tmp57);
      if(tmp58 > 0.5)
      {
        tmp58 -= 1.0;
        tmp57 += 1.0;
      }
      else if(tmp58 < -0.5)
      {
        tmp58 += 1.0;
        tmp57 -= 1.0;
      }
      if(fabs(tmp58) < 1e-10 && ((unsigned long)tmp57 & 1))
      {
        tmp54 = -pow(-tmp52, tmp55)*pow(tmp52, tmp56);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp52, tmp53);
      }
    }
  }
  else
  {
    tmp54 = pow(tmp52, tmp53);
  }
  if(isnan(tmp54) || isinf(tmp54))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp52, tmp53);
  }tmp59 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp54,"(r_init[495] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp59 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[495] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp59);
    }
  }
  (data->simulationInfo->realParameter[999] /* omega_c[495] PARAM */) = sqrt(tmp59);
  TRACE_POP
}

/*
equation index: 13013
type: SIMPLE_ASSIGN
r_init[494] = r_min + 494.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13013};
  (data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (494.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13014
type: SIMPLE_ASSIGN
omega_c[494] = sqrt(G * Md / (r_init[494] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13014(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13014};
  modelica_real tmp60;
  modelica_real tmp61;
  modelica_real tmp62;
  modelica_real tmp63;
  modelica_real tmp64;
  modelica_real tmp65;
  modelica_real tmp66;
  modelica_real tmp67;
  modelica_real tmp68;
  modelica_real tmp69;
  tmp60 = (data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */);
  tmp61 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp62 = (tmp60 * tmp60) + (tmp61 * tmp61);
  tmp63 = 1.5;
  if(tmp62 < 0.0 && tmp63 != 0.0)
  {
    tmp65 = modf(tmp63, &tmp66);
    
    if(tmp65 > 0.5)
    {
      tmp65 -= 1.0;
      tmp66 += 1.0;
    }
    else if(tmp65 < -0.5)
    {
      tmp65 += 1.0;
      tmp66 -= 1.0;
    }
    
    if(fabs(tmp65) < 1e-10)
      tmp64 = pow(tmp62, tmp66);
    else
    {
      tmp68 = modf(1.0/tmp63, &tmp67);
      if(tmp68 > 0.5)
      {
        tmp68 -= 1.0;
        tmp67 += 1.0;
      }
      else if(tmp68 < -0.5)
      {
        tmp68 += 1.0;
        tmp67 -= 1.0;
      }
      if(fabs(tmp68) < 1e-10 && ((unsigned long)tmp67 & 1))
      {
        tmp64 = -pow(-tmp62, tmp65)*pow(tmp62, tmp66);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp62, tmp63);
      }
    }
  }
  else
  {
    tmp64 = pow(tmp62, tmp63);
  }
  if(isnan(tmp64) || isinf(tmp64))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp62, tmp63);
  }tmp69 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp64,"(r_init[494] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp69 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[494] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp69);
    }
  }
  (data->simulationInfo->realParameter[998] /* omega_c[494] PARAM */) = sqrt(tmp69);
  TRACE_POP
}

/*
equation index: 13015
type: SIMPLE_ASSIGN
r_init[493] = r_min + 493.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13015};
  (data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (493.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13016
type: SIMPLE_ASSIGN
omega_c[493] = sqrt(G * Md / (r_init[493] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13016(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13016};
  modelica_real tmp70;
  modelica_real tmp71;
  modelica_real tmp72;
  modelica_real tmp73;
  modelica_real tmp74;
  modelica_real tmp75;
  modelica_real tmp76;
  modelica_real tmp77;
  modelica_real tmp78;
  modelica_real tmp79;
  tmp70 = (data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */);
  tmp71 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp72 = (tmp70 * tmp70) + (tmp71 * tmp71);
  tmp73 = 1.5;
  if(tmp72 < 0.0 && tmp73 != 0.0)
  {
    tmp75 = modf(tmp73, &tmp76);
    
    if(tmp75 > 0.5)
    {
      tmp75 -= 1.0;
      tmp76 += 1.0;
    }
    else if(tmp75 < -0.5)
    {
      tmp75 += 1.0;
      tmp76 -= 1.0;
    }
    
    if(fabs(tmp75) < 1e-10)
      tmp74 = pow(tmp72, tmp76);
    else
    {
      tmp78 = modf(1.0/tmp73, &tmp77);
      if(tmp78 > 0.5)
      {
        tmp78 -= 1.0;
        tmp77 += 1.0;
      }
      else if(tmp78 < -0.5)
      {
        tmp78 += 1.0;
        tmp77 -= 1.0;
      }
      if(fabs(tmp78) < 1e-10 && ((unsigned long)tmp77 & 1))
      {
        tmp74 = -pow(-tmp72, tmp75)*pow(tmp72, tmp76);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp72, tmp73);
      }
    }
  }
  else
  {
    tmp74 = pow(tmp72, tmp73);
  }
  if(isnan(tmp74) || isinf(tmp74))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp72, tmp73);
  }tmp79 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp74,"(r_init[493] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp79 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[493] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp79);
    }
  }
  (data->simulationInfo->realParameter[997] /* omega_c[493] PARAM */) = sqrt(tmp79);
  TRACE_POP
}

/*
equation index: 13017
type: SIMPLE_ASSIGN
r_init[492] = r_min + 492.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13017(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13017};
  (data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (492.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13018
type: SIMPLE_ASSIGN
omega_c[492] = sqrt(G * Md / (r_init[492] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13018(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13018};
  modelica_real tmp80;
  modelica_real tmp81;
  modelica_real tmp82;
  modelica_real tmp83;
  modelica_real tmp84;
  modelica_real tmp85;
  modelica_real tmp86;
  modelica_real tmp87;
  modelica_real tmp88;
  modelica_real tmp89;
  tmp80 = (data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */);
  tmp81 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp82 = (tmp80 * tmp80) + (tmp81 * tmp81);
  tmp83 = 1.5;
  if(tmp82 < 0.0 && tmp83 != 0.0)
  {
    tmp85 = modf(tmp83, &tmp86);
    
    if(tmp85 > 0.5)
    {
      tmp85 -= 1.0;
      tmp86 += 1.0;
    }
    else if(tmp85 < -0.5)
    {
      tmp85 += 1.0;
      tmp86 -= 1.0;
    }
    
    if(fabs(tmp85) < 1e-10)
      tmp84 = pow(tmp82, tmp86);
    else
    {
      tmp88 = modf(1.0/tmp83, &tmp87);
      if(tmp88 > 0.5)
      {
        tmp88 -= 1.0;
        tmp87 += 1.0;
      }
      else if(tmp88 < -0.5)
      {
        tmp88 += 1.0;
        tmp87 -= 1.0;
      }
      if(fabs(tmp88) < 1e-10 && ((unsigned long)tmp87 & 1))
      {
        tmp84 = -pow(-tmp82, tmp85)*pow(tmp82, tmp86);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp82, tmp83);
      }
    }
  }
  else
  {
    tmp84 = pow(tmp82, tmp83);
  }
  if(isnan(tmp84) || isinf(tmp84))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp82, tmp83);
  }tmp89 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp84,"(r_init[492] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp89 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[492] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp89);
    }
  }
  (data->simulationInfo->realParameter[996] /* omega_c[492] PARAM */) = sqrt(tmp89);
  TRACE_POP
}

/*
equation index: 13019
type: SIMPLE_ASSIGN
r_init[491] = r_min + 491.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13019};
  (data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (491.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13020
type: SIMPLE_ASSIGN
omega_c[491] = sqrt(G * Md / (r_init[491] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13020(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13020};
  modelica_real tmp90;
  modelica_real tmp91;
  modelica_real tmp92;
  modelica_real tmp93;
  modelica_real tmp94;
  modelica_real tmp95;
  modelica_real tmp96;
  modelica_real tmp97;
  modelica_real tmp98;
  modelica_real tmp99;
  tmp90 = (data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */);
  tmp91 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp92 = (tmp90 * tmp90) + (tmp91 * tmp91);
  tmp93 = 1.5;
  if(tmp92 < 0.0 && tmp93 != 0.0)
  {
    tmp95 = modf(tmp93, &tmp96);
    
    if(tmp95 > 0.5)
    {
      tmp95 -= 1.0;
      tmp96 += 1.0;
    }
    else if(tmp95 < -0.5)
    {
      tmp95 += 1.0;
      tmp96 -= 1.0;
    }
    
    if(fabs(tmp95) < 1e-10)
      tmp94 = pow(tmp92, tmp96);
    else
    {
      tmp98 = modf(1.0/tmp93, &tmp97);
      if(tmp98 > 0.5)
      {
        tmp98 -= 1.0;
        tmp97 += 1.0;
      }
      else if(tmp98 < -0.5)
      {
        tmp98 += 1.0;
        tmp97 -= 1.0;
      }
      if(fabs(tmp98) < 1e-10 && ((unsigned long)tmp97 & 1))
      {
        tmp94 = -pow(-tmp92, tmp95)*pow(tmp92, tmp96);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp92, tmp93);
      }
    }
  }
  else
  {
    tmp94 = pow(tmp92, tmp93);
  }
  if(isnan(tmp94) || isinf(tmp94))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp92, tmp93);
  }tmp99 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp94,"(r_init[491] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp99 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[491] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp99);
    }
  }
  (data->simulationInfo->realParameter[995] /* omega_c[491] PARAM */) = sqrt(tmp99);
  TRACE_POP
}

/*
equation index: 13021
type: SIMPLE_ASSIGN
r_init[490] = r_min + 490.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13021};
  (data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (490.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13022
type: SIMPLE_ASSIGN
omega_c[490] = sqrt(G * Md / (r_init[490] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13022(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13022};
  modelica_real tmp100;
  modelica_real tmp101;
  modelica_real tmp102;
  modelica_real tmp103;
  modelica_real tmp104;
  modelica_real tmp105;
  modelica_real tmp106;
  modelica_real tmp107;
  modelica_real tmp108;
  modelica_real tmp109;
  tmp100 = (data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */);
  tmp101 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp102 = (tmp100 * tmp100) + (tmp101 * tmp101);
  tmp103 = 1.5;
  if(tmp102 < 0.0 && tmp103 != 0.0)
  {
    tmp105 = modf(tmp103, &tmp106);
    
    if(tmp105 > 0.5)
    {
      tmp105 -= 1.0;
      tmp106 += 1.0;
    }
    else if(tmp105 < -0.5)
    {
      tmp105 += 1.0;
      tmp106 -= 1.0;
    }
    
    if(fabs(tmp105) < 1e-10)
      tmp104 = pow(tmp102, tmp106);
    else
    {
      tmp108 = modf(1.0/tmp103, &tmp107);
      if(tmp108 > 0.5)
      {
        tmp108 -= 1.0;
        tmp107 += 1.0;
      }
      else if(tmp108 < -0.5)
      {
        tmp108 += 1.0;
        tmp107 -= 1.0;
      }
      if(fabs(tmp108) < 1e-10 && ((unsigned long)tmp107 & 1))
      {
        tmp104 = -pow(-tmp102, tmp105)*pow(tmp102, tmp106);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp102, tmp103);
      }
    }
  }
  else
  {
    tmp104 = pow(tmp102, tmp103);
  }
  if(isnan(tmp104) || isinf(tmp104))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp102, tmp103);
  }tmp109 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp104,"(r_init[490] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp109 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[490] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp109);
    }
  }
  (data->simulationInfo->realParameter[994] /* omega_c[490] PARAM */) = sqrt(tmp109);
  TRACE_POP
}

/*
equation index: 13023
type: SIMPLE_ASSIGN
r_init[489] = r_min + 489.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13023};
  (data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (489.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13024
type: SIMPLE_ASSIGN
omega_c[489] = sqrt(G * Md / (r_init[489] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13024(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13024};
  modelica_real tmp110;
  modelica_real tmp111;
  modelica_real tmp112;
  modelica_real tmp113;
  modelica_real tmp114;
  modelica_real tmp115;
  modelica_real tmp116;
  modelica_real tmp117;
  modelica_real tmp118;
  modelica_real tmp119;
  tmp110 = (data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */);
  tmp111 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp112 = (tmp110 * tmp110) + (tmp111 * tmp111);
  tmp113 = 1.5;
  if(tmp112 < 0.0 && tmp113 != 0.0)
  {
    tmp115 = modf(tmp113, &tmp116);
    
    if(tmp115 > 0.5)
    {
      tmp115 -= 1.0;
      tmp116 += 1.0;
    }
    else if(tmp115 < -0.5)
    {
      tmp115 += 1.0;
      tmp116 -= 1.0;
    }
    
    if(fabs(tmp115) < 1e-10)
      tmp114 = pow(tmp112, tmp116);
    else
    {
      tmp118 = modf(1.0/tmp113, &tmp117);
      if(tmp118 > 0.5)
      {
        tmp118 -= 1.0;
        tmp117 += 1.0;
      }
      else if(tmp118 < -0.5)
      {
        tmp118 += 1.0;
        tmp117 -= 1.0;
      }
      if(fabs(tmp118) < 1e-10 && ((unsigned long)tmp117 & 1))
      {
        tmp114 = -pow(-tmp112, tmp115)*pow(tmp112, tmp116);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp112, tmp113);
      }
    }
  }
  else
  {
    tmp114 = pow(tmp112, tmp113);
  }
  if(isnan(tmp114) || isinf(tmp114))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp112, tmp113);
  }tmp119 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp114,"(r_init[489] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp119 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[489] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp119);
    }
  }
  (data->simulationInfo->realParameter[993] /* omega_c[489] PARAM */) = sqrt(tmp119);
  TRACE_POP
}

/*
equation index: 13025
type: SIMPLE_ASSIGN
r_init[488] = r_min + 488.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13025};
  (data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (488.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13026
type: SIMPLE_ASSIGN
omega_c[488] = sqrt(G * Md / (r_init[488] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13026(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13026};
  modelica_real tmp120;
  modelica_real tmp121;
  modelica_real tmp122;
  modelica_real tmp123;
  modelica_real tmp124;
  modelica_real tmp125;
  modelica_real tmp126;
  modelica_real tmp127;
  modelica_real tmp128;
  modelica_real tmp129;
  tmp120 = (data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */);
  tmp121 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp122 = (tmp120 * tmp120) + (tmp121 * tmp121);
  tmp123 = 1.5;
  if(tmp122 < 0.0 && tmp123 != 0.0)
  {
    tmp125 = modf(tmp123, &tmp126);
    
    if(tmp125 > 0.5)
    {
      tmp125 -= 1.0;
      tmp126 += 1.0;
    }
    else if(tmp125 < -0.5)
    {
      tmp125 += 1.0;
      tmp126 -= 1.0;
    }
    
    if(fabs(tmp125) < 1e-10)
      tmp124 = pow(tmp122, tmp126);
    else
    {
      tmp128 = modf(1.0/tmp123, &tmp127);
      if(tmp128 > 0.5)
      {
        tmp128 -= 1.0;
        tmp127 += 1.0;
      }
      else if(tmp128 < -0.5)
      {
        tmp128 += 1.0;
        tmp127 -= 1.0;
      }
      if(fabs(tmp128) < 1e-10 && ((unsigned long)tmp127 & 1))
      {
        tmp124 = -pow(-tmp122, tmp125)*pow(tmp122, tmp126);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp122, tmp123);
      }
    }
  }
  else
  {
    tmp124 = pow(tmp122, tmp123);
  }
  if(isnan(tmp124) || isinf(tmp124))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp122, tmp123);
  }tmp129 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp124,"(r_init[488] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp129 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[488] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp129);
    }
  }
  (data->simulationInfo->realParameter[992] /* omega_c[488] PARAM */) = sqrt(tmp129);
  TRACE_POP
}

/*
equation index: 13027
type: SIMPLE_ASSIGN
r_init[487] = r_min + 487.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13027(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13027};
  (data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (487.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13028
type: SIMPLE_ASSIGN
omega_c[487] = sqrt(G * Md / (r_init[487] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13028};
  modelica_real tmp130;
  modelica_real tmp131;
  modelica_real tmp132;
  modelica_real tmp133;
  modelica_real tmp134;
  modelica_real tmp135;
  modelica_real tmp136;
  modelica_real tmp137;
  modelica_real tmp138;
  modelica_real tmp139;
  tmp130 = (data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */);
  tmp131 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp132 = (tmp130 * tmp130) + (tmp131 * tmp131);
  tmp133 = 1.5;
  if(tmp132 < 0.0 && tmp133 != 0.0)
  {
    tmp135 = modf(tmp133, &tmp136);
    
    if(tmp135 > 0.5)
    {
      tmp135 -= 1.0;
      tmp136 += 1.0;
    }
    else if(tmp135 < -0.5)
    {
      tmp135 += 1.0;
      tmp136 -= 1.0;
    }
    
    if(fabs(tmp135) < 1e-10)
      tmp134 = pow(tmp132, tmp136);
    else
    {
      tmp138 = modf(1.0/tmp133, &tmp137);
      if(tmp138 > 0.5)
      {
        tmp138 -= 1.0;
        tmp137 += 1.0;
      }
      else if(tmp138 < -0.5)
      {
        tmp138 += 1.0;
        tmp137 -= 1.0;
      }
      if(fabs(tmp138) < 1e-10 && ((unsigned long)tmp137 & 1))
      {
        tmp134 = -pow(-tmp132, tmp135)*pow(tmp132, tmp136);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp132, tmp133);
      }
    }
  }
  else
  {
    tmp134 = pow(tmp132, tmp133);
  }
  if(isnan(tmp134) || isinf(tmp134))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp132, tmp133);
  }tmp139 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp134,"(r_init[487] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp139 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[487] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp139);
    }
  }
  (data->simulationInfo->realParameter[991] /* omega_c[487] PARAM */) = sqrt(tmp139);
  TRACE_POP
}

/*
equation index: 13029
type: SIMPLE_ASSIGN
r_init[486] = r_min + 486.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13029};
  (data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (486.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13030
type: SIMPLE_ASSIGN
omega_c[486] = sqrt(G * Md / (r_init[486] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13030(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13030};
  modelica_real tmp140;
  modelica_real tmp141;
  modelica_real tmp142;
  modelica_real tmp143;
  modelica_real tmp144;
  modelica_real tmp145;
  modelica_real tmp146;
  modelica_real tmp147;
  modelica_real tmp148;
  modelica_real tmp149;
  tmp140 = (data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */);
  tmp141 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp142 = (tmp140 * tmp140) + (tmp141 * tmp141);
  tmp143 = 1.5;
  if(tmp142 < 0.0 && tmp143 != 0.0)
  {
    tmp145 = modf(tmp143, &tmp146);
    
    if(tmp145 > 0.5)
    {
      tmp145 -= 1.0;
      tmp146 += 1.0;
    }
    else if(tmp145 < -0.5)
    {
      tmp145 += 1.0;
      tmp146 -= 1.0;
    }
    
    if(fabs(tmp145) < 1e-10)
      tmp144 = pow(tmp142, tmp146);
    else
    {
      tmp148 = modf(1.0/tmp143, &tmp147);
      if(tmp148 > 0.5)
      {
        tmp148 -= 1.0;
        tmp147 += 1.0;
      }
      else if(tmp148 < -0.5)
      {
        tmp148 += 1.0;
        tmp147 -= 1.0;
      }
      if(fabs(tmp148) < 1e-10 && ((unsigned long)tmp147 & 1))
      {
        tmp144 = -pow(-tmp142, tmp145)*pow(tmp142, tmp146);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp142, tmp143);
      }
    }
  }
  else
  {
    tmp144 = pow(tmp142, tmp143);
  }
  if(isnan(tmp144) || isinf(tmp144))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp142, tmp143);
  }tmp149 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp144,"(r_init[486] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp149 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[486] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp149);
    }
  }
  (data->simulationInfo->realParameter[990] /* omega_c[486] PARAM */) = sqrt(tmp149);
  TRACE_POP
}

/*
equation index: 13031
type: SIMPLE_ASSIGN
r_init[485] = r_min + 485.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13031};
  (data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (485.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13032
type: SIMPLE_ASSIGN
omega_c[485] = sqrt(G * Md / (r_init[485] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13032(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13032};
  modelica_real tmp150;
  modelica_real tmp151;
  modelica_real tmp152;
  modelica_real tmp153;
  modelica_real tmp154;
  modelica_real tmp155;
  modelica_real tmp156;
  modelica_real tmp157;
  modelica_real tmp158;
  modelica_real tmp159;
  tmp150 = (data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */);
  tmp151 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp152 = (tmp150 * tmp150) + (tmp151 * tmp151);
  tmp153 = 1.5;
  if(tmp152 < 0.0 && tmp153 != 0.0)
  {
    tmp155 = modf(tmp153, &tmp156);
    
    if(tmp155 > 0.5)
    {
      tmp155 -= 1.0;
      tmp156 += 1.0;
    }
    else if(tmp155 < -0.5)
    {
      tmp155 += 1.0;
      tmp156 -= 1.0;
    }
    
    if(fabs(tmp155) < 1e-10)
      tmp154 = pow(tmp152, tmp156);
    else
    {
      tmp158 = modf(1.0/tmp153, &tmp157);
      if(tmp158 > 0.5)
      {
        tmp158 -= 1.0;
        tmp157 += 1.0;
      }
      else if(tmp158 < -0.5)
      {
        tmp158 += 1.0;
        tmp157 -= 1.0;
      }
      if(fabs(tmp158) < 1e-10 && ((unsigned long)tmp157 & 1))
      {
        tmp154 = -pow(-tmp152, tmp155)*pow(tmp152, tmp156);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp152, tmp153);
      }
    }
  }
  else
  {
    tmp154 = pow(tmp152, tmp153);
  }
  if(isnan(tmp154) || isinf(tmp154))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp152, tmp153);
  }tmp159 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp154,"(r_init[485] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp159 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[485] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp159);
    }
  }
  (data->simulationInfo->realParameter[989] /* omega_c[485] PARAM */) = sqrt(tmp159);
  TRACE_POP
}

/*
equation index: 13033
type: SIMPLE_ASSIGN
r_init[484] = r_min + 484.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13033(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13033};
  (data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (484.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13034
type: SIMPLE_ASSIGN
omega_c[484] = sqrt(G * Md / (r_init[484] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13034(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13034};
  modelica_real tmp160;
  modelica_real tmp161;
  modelica_real tmp162;
  modelica_real tmp163;
  modelica_real tmp164;
  modelica_real tmp165;
  modelica_real tmp166;
  modelica_real tmp167;
  modelica_real tmp168;
  modelica_real tmp169;
  tmp160 = (data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */);
  tmp161 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp162 = (tmp160 * tmp160) + (tmp161 * tmp161);
  tmp163 = 1.5;
  if(tmp162 < 0.0 && tmp163 != 0.0)
  {
    tmp165 = modf(tmp163, &tmp166);
    
    if(tmp165 > 0.5)
    {
      tmp165 -= 1.0;
      tmp166 += 1.0;
    }
    else if(tmp165 < -0.5)
    {
      tmp165 += 1.0;
      tmp166 -= 1.0;
    }
    
    if(fabs(tmp165) < 1e-10)
      tmp164 = pow(tmp162, tmp166);
    else
    {
      tmp168 = modf(1.0/tmp163, &tmp167);
      if(tmp168 > 0.5)
      {
        tmp168 -= 1.0;
        tmp167 += 1.0;
      }
      else if(tmp168 < -0.5)
      {
        tmp168 += 1.0;
        tmp167 -= 1.0;
      }
      if(fabs(tmp168) < 1e-10 && ((unsigned long)tmp167 & 1))
      {
        tmp164 = -pow(-tmp162, tmp165)*pow(tmp162, tmp166);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp162, tmp163);
      }
    }
  }
  else
  {
    tmp164 = pow(tmp162, tmp163);
  }
  if(isnan(tmp164) || isinf(tmp164))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp162, tmp163);
  }tmp169 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp164,"(r_init[484] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp169 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[484] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp169);
    }
  }
  (data->simulationInfo->realParameter[988] /* omega_c[484] PARAM */) = sqrt(tmp169);
  TRACE_POP
}

/*
equation index: 13035
type: SIMPLE_ASSIGN
r_init[483] = r_min + 483.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13035};
  (data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (483.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13036
type: SIMPLE_ASSIGN
omega_c[483] = sqrt(G * Md / (r_init[483] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13036(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13036};
  modelica_real tmp170;
  modelica_real tmp171;
  modelica_real tmp172;
  modelica_real tmp173;
  modelica_real tmp174;
  modelica_real tmp175;
  modelica_real tmp176;
  modelica_real tmp177;
  modelica_real tmp178;
  modelica_real tmp179;
  tmp170 = (data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */);
  tmp171 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp172 = (tmp170 * tmp170) + (tmp171 * tmp171);
  tmp173 = 1.5;
  if(tmp172 < 0.0 && tmp173 != 0.0)
  {
    tmp175 = modf(tmp173, &tmp176);
    
    if(tmp175 > 0.5)
    {
      tmp175 -= 1.0;
      tmp176 += 1.0;
    }
    else if(tmp175 < -0.5)
    {
      tmp175 += 1.0;
      tmp176 -= 1.0;
    }
    
    if(fabs(tmp175) < 1e-10)
      tmp174 = pow(tmp172, tmp176);
    else
    {
      tmp178 = modf(1.0/tmp173, &tmp177);
      if(tmp178 > 0.5)
      {
        tmp178 -= 1.0;
        tmp177 += 1.0;
      }
      else if(tmp178 < -0.5)
      {
        tmp178 += 1.0;
        tmp177 -= 1.0;
      }
      if(fabs(tmp178) < 1e-10 && ((unsigned long)tmp177 & 1))
      {
        tmp174 = -pow(-tmp172, tmp175)*pow(tmp172, tmp176);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp172, tmp173);
      }
    }
  }
  else
  {
    tmp174 = pow(tmp172, tmp173);
  }
  if(isnan(tmp174) || isinf(tmp174))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp172, tmp173);
  }tmp179 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp174,"(r_init[483] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp179 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[483] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp179);
    }
  }
  (data->simulationInfo->realParameter[987] /* omega_c[483] PARAM */) = sqrt(tmp179);
  TRACE_POP
}

/*
equation index: 13037
type: SIMPLE_ASSIGN
r_init[482] = r_min + 482.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13037};
  (data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (482.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13038
type: SIMPLE_ASSIGN
omega_c[482] = sqrt(G * Md / (r_init[482] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13038(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13038};
  modelica_real tmp180;
  modelica_real tmp181;
  modelica_real tmp182;
  modelica_real tmp183;
  modelica_real tmp184;
  modelica_real tmp185;
  modelica_real tmp186;
  modelica_real tmp187;
  modelica_real tmp188;
  modelica_real tmp189;
  tmp180 = (data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */);
  tmp181 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp182 = (tmp180 * tmp180) + (tmp181 * tmp181);
  tmp183 = 1.5;
  if(tmp182 < 0.0 && tmp183 != 0.0)
  {
    tmp185 = modf(tmp183, &tmp186);
    
    if(tmp185 > 0.5)
    {
      tmp185 -= 1.0;
      tmp186 += 1.0;
    }
    else if(tmp185 < -0.5)
    {
      tmp185 += 1.0;
      tmp186 -= 1.0;
    }
    
    if(fabs(tmp185) < 1e-10)
      tmp184 = pow(tmp182, tmp186);
    else
    {
      tmp188 = modf(1.0/tmp183, &tmp187);
      if(tmp188 > 0.5)
      {
        tmp188 -= 1.0;
        tmp187 += 1.0;
      }
      else if(tmp188 < -0.5)
      {
        tmp188 += 1.0;
        tmp187 -= 1.0;
      }
      if(fabs(tmp188) < 1e-10 && ((unsigned long)tmp187 & 1))
      {
        tmp184 = -pow(-tmp182, tmp185)*pow(tmp182, tmp186);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp182, tmp183);
      }
    }
  }
  else
  {
    tmp184 = pow(tmp182, tmp183);
  }
  if(isnan(tmp184) || isinf(tmp184))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp182, tmp183);
  }tmp189 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp184,"(r_init[482] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp189 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[482] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp189);
    }
  }
  (data->simulationInfo->realParameter[986] /* omega_c[482] PARAM */) = sqrt(tmp189);
  TRACE_POP
}

/*
equation index: 13039
type: SIMPLE_ASSIGN
r_init[481] = r_min + 481.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13039};
  (data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (481.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13040
type: SIMPLE_ASSIGN
omega_c[481] = sqrt(G * Md / (r_init[481] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13040(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13040};
  modelica_real tmp190;
  modelica_real tmp191;
  modelica_real tmp192;
  modelica_real tmp193;
  modelica_real tmp194;
  modelica_real tmp195;
  modelica_real tmp196;
  modelica_real tmp197;
  modelica_real tmp198;
  modelica_real tmp199;
  tmp190 = (data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */);
  tmp191 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp192 = (tmp190 * tmp190) + (tmp191 * tmp191);
  tmp193 = 1.5;
  if(tmp192 < 0.0 && tmp193 != 0.0)
  {
    tmp195 = modf(tmp193, &tmp196);
    
    if(tmp195 > 0.5)
    {
      tmp195 -= 1.0;
      tmp196 += 1.0;
    }
    else if(tmp195 < -0.5)
    {
      tmp195 += 1.0;
      tmp196 -= 1.0;
    }
    
    if(fabs(tmp195) < 1e-10)
      tmp194 = pow(tmp192, tmp196);
    else
    {
      tmp198 = modf(1.0/tmp193, &tmp197);
      if(tmp198 > 0.5)
      {
        tmp198 -= 1.0;
        tmp197 += 1.0;
      }
      else if(tmp198 < -0.5)
      {
        tmp198 += 1.0;
        tmp197 -= 1.0;
      }
      if(fabs(tmp198) < 1e-10 && ((unsigned long)tmp197 & 1))
      {
        tmp194 = -pow(-tmp192, tmp195)*pow(tmp192, tmp196);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp192, tmp193);
      }
    }
  }
  else
  {
    tmp194 = pow(tmp192, tmp193);
  }
  if(isnan(tmp194) || isinf(tmp194))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp192, tmp193);
  }tmp199 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp194,"(r_init[481] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp199 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[481] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp199);
    }
  }
  (data->simulationInfo->realParameter[985] /* omega_c[481] PARAM */) = sqrt(tmp199);
  TRACE_POP
}

/*
equation index: 13041
type: SIMPLE_ASSIGN
r_init[480] = r_min + 480.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13041};
  (data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (480.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13042
type: SIMPLE_ASSIGN
omega_c[480] = sqrt(G * Md / (r_init[480] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13042(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13042};
  modelica_real tmp200;
  modelica_real tmp201;
  modelica_real tmp202;
  modelica_real tmp203;
  modelica_real tmp204;
  modelica_real tmp205;
  modelica_real tmp206;
  modelica_real tmp207;
  modelica_real tmp208;
  modelica_real tmp209;
  tmp200 = (data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */);
  tmp201 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp202 = (tmp200 * tmp200) + (tmp201 * tmp201);
  tmp203 = 1.5;
  if(tmp202 < 0.0 && tmp203 != 0.0)
  {
    tmp205 = modf(tmp203, &tmp206);
    
    if(tmp205 > 0.5)
    {
      tmp205 -= 1.0;
      tmp206 += 1.0;
    }
    else if(tmp205 < -0.5)
    {
      tmp205 += 1.0;
      tmp206 -= 1.0;
    }
    
    if(fabs(tmp205) < 1e-10)
      tmp204 = pow(tmp202, tmp206);
    else
    {
      tmp208 = modf(1.0/tmp203, &tmp207);
      if(tmp208 > 0.5)
      {
        tmp208 -= 1.0;
        tmp207 += 1.0;
      }
      else if(tmp208 < -0.5)
      {
        tmp208 += 1.0;
        tmp207 -= 1.0;
      }
      if(fabs(tmp208) < 1e-10 && ((unsigned long)tmp207 & 1))
      {
        tmp204 = -pow(-tmp202, tmp205)*pow(tmp202, tmp206);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp202, tmp203);
      }
    }
  }
  else
  {
    tmp204 = pow(tmp202, tmp203);
  }
  if(isnan(tmp204) || isinf(tmp204))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp202, tmp203);
  }tmp209 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp204,"(r_init[480] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp209 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[480] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp209);
    }
  }
  (data->simulationInfo->realParameter[984] /* omega_c[480] PARAM */) = sqrt(tmp209);
  TRACE_POP
}

/*
equation index: 13043
type: SIMPLE_ASSIGN
r_init[479] = r_min + 479.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13043(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13043};
  (data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (479.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13044
type: SIMPLE_ASSIGN
omega_c[479] = sqrt(G * Md / (r_init[479] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13044};
  modelica_real tmp210;
  modelica_real tmp211;
  modelica_real tmp212;
  modelica_real tmp213;
  modelica_real tmp214;
  modelica_real tmp215;
  modelica_real tmp216;
  modelica_real tmp217;
  modelica_real tmp218;
  modelica_real tmp219;
  tmp210 = (data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */);
  tmp211 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp212 = (tmp210 * tmp210) + (tmp211 * tmp211);
  tmp213 = 1.5;
  if(tmp212 < 0.0 && tmp213 != 0.0)
  {
    tmp215 = modf(tmp213, &tmp216);
    
    if(tmp215 > 0.5)
    {
      tmp215 -= 1.0;
      tmp216 += 1.0;
    }
    else if(tmp215 < -0.5)
    {
      tmp215 += 1.0;
      tmp216 -= 1.0;
    }
    
    if(fabs(tmp215) < 1e-10)
      tmp214 = pow(tmp212, tmp216);
    else
    {
      tmp218 = modf(1.0/tmp213, &tmp217);
      if(tmp218 > 0.5)
      {
        tmp218 -= 1.0;
        tmp217 += 1.0;
      }
      else if(tmp218 < -0.5)
      {
        tmp218 += 1.0;
        tmp217 -= 1.0;
      }
      if(fabs(tmp218) < 1e-10 && ((unsigned long)tmp217 & 1))
      {
        tmp214 = -pow(-tmp212, tmp215)*pow(tmp212, tmp216);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp212, tmp213);
      }
    }
  }
  else
  {
    tmp214 = pow(tmp212, tmp213);
  }
  if(isnan(tmp214) || isinf(tmp214))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp212, tmp213);
  }tmp219 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp214,"(r_init[479] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp219 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[479] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp219);
    }
  }
  (data->simulationInfo->realParameter[983] /* omega_c[479] PARAM */) = sqrt(tmp219);
  TRACE_POP
}

/*
equation index: 13045
type: SIMPLE_ASSIGN
r_init[478] = r_min + 478.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13045};
  (data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (478.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13046
type: SIMPLE_ASSIGN
omega_c[478] = sqrt(G * Md / (r_init[478] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13046(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13046};
  modelica_real tmp220;
  modelica_real tmp221;
  modelica_real tmp222;
  modelica_real tmp223;
  modelica_real tmp224;
  modelica_real tmp225;
  modelica_real tmp226;
  modelica_real tmp227;
  modelica_real tmp228;
  modelica_real tmp229;
  tmp220 = (data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */);
  tmp221 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp222 = (tmp220 * tmp220) + (tmp221 * tmp221);
  tmp223 = 1.5;
  if(tmp222 < 0.0 && tmp223 != 0.0)
  {
    tmp225 = modf(tmp223, &tmp226);
    
    if(tmp225 > 0.5)
    {
      tmp225 -= 1.0;
      tmp226 += 1.0;
    }
    else if(tmp225 < -0.5)
    {
      tmp225 += 1.0;
      tmp226 -= 1.0;
    }
    
    if(fabs(tmp225) < 1e-10)
      tmp224 = pow(tmp222, tmp226);
    else
    {
      tmp228 = modf(1.0/tmp223, &tmp227);
      if(tmp228 > 0.5)
      {
        tmp228 -= 1.0;
        tmp227 += 1.0;
      }
      else if(tmp228 < -0.5)
      {
        tmp228 += 1.0;
        tmp227 -= 1.0;
      }
      if(fabs(tmp228) < 1e-10 && ((unsigned long)tmp227 & 1))
      {
        tmp224 = -pow(-tmp222, tmp225)*pow(tmp222, tmp226);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp222, tmp223);
      }
    }
  }
  else
  {
    tmp224 = pow(tmp222, tmp223);
  }
  if(isnan(tmp224) || isinf(tmp224))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp222, tmp223);
  }tmp229 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp224,"(r_init[478] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp229 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[478] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp229);
    }
  }
  (data->simulationInfo->realParameter[982] /* omega_c[478] PARAM */) = sqrt(tmp229);
  TRACE_POP
}

/*
equation index: 13047
type: SIMPLE_ASSIGN
r_init[477] = r_min + 477.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13047};
  (data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (477.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13048
type: SIMPLE_ASSIGN
omega_c[477] = sqrt(G * Md / (r_init[477] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13048(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13048};
  modelica_real tmp230;
  modelica_real tmp231;
  modelica_real tmp232;
  modelica_real tmp233;
  modelica_real tmp234;
  modelica_real tmp235;
  modelica_real tmp236;
  modelica_real tmp237;
  modelica_real tmp238;
  modelica_real tmp239;
  tmp230 = (data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */);
  tmp231 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp232 = (tmp230 * tmp230) + (tmp231 * tmp231);
  tmp233 = 1.5;
  if(tmp232 < 0.0 && tmp233 != 0.0)
  {
    tmp235 = modf(tmp233, &tmp236);
    
    if(tmp235 > 0.5)
    {
      tmp235 -= 1.0;
      tmp236 += 1.0;
    }
    else if(tmp235 < -0.5)
    {
      tmp235 += 1.0;
      tmp236 -= 1.0;
    }
    
    if(fabs(tmp235) < 1e-10)
      tmp234 = pow(tmp232, tmp236);
    else
    {
      tmp238 = modf(1.0/tmp233, &tmp237);
      if(tmp238 > 0.5)
      {
        tmp238 -= 1.0;
        tmp237 += 1.0;
      }
      else if(tmp238 < -0.5)
      {
        tmp238 += 1.0;
        tmp237 -= 1.0;
      }
      if(fabs(tmp238) < 1e-10 && ((unsigned long)tmp237 & 1))
      {
        tmp234 = -pow(-tmp232, tmp235)*pow(tmp232, tmp236);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp232, tmp233);
      }
    }
  }
  else
  {
    tmp234 = pow(tmp232, tmp233);
  }
  if(isnan(tmp234) || isinf(tmp234))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp232, tmp233);
  }tmp239 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp234,"(r_init[477] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp239 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[477] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp239);
    }
  }
  (data->simulationInfo->realParameter[981] /* omega_c[477] PARAM */) = sqrt(tmp239);
  TRACE_POP
}

/*
equation index: 13049
type: SIMPLE_ASSIGN
r_init[476] = r_min + 476.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13049(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13049};
  (data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (476.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13050
type: SIMPLE_ASSIGN
omega_c[476] = sqrt(G * Md / (r_init[476] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13050(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13050};
  modelica_real tmp240;
  modelica_real tmp241;
  modelica_real tmp242;
  modelica_real tmp243;
  modelica_real tmp244;
  modelica_real tmp245;
  modelica_real tmp246;
  modelica_real tmp247;
  modelica_real tmp248;
  modelica_real tmp249;
  tmp240 = (data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */);
  tmp241 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp242 = (tmp240 * tmp240) + (tmp241 * tmp241);
  tmp243 = 1.5;
  if(tmp242 < 0.0 && tmp243 != 0.0)
  {
    tmp245 = modf(tmp243, &tmp246);
    
    if(tmp245 > 0.5)
    {
      tmp245 -= 1.0;
      tmp246 += 1.0;
    }
    else if(tmp245 < -0.5)
    {
      tmp245 += 1.0;
      tmp246 -= 1.0;
    }
    
    if(fabs(tmp245) < 1e-10)
      tmp244 = pow(tmp242, tmp246);
    else
    {
      tmp248 = modf(1.0/tmp243, &tmp247);
      if(tmp248 > 0.5)
      {
        tmp248 -= 1.0;
        tmp247 += 1.0;
      }
      else if(tmp248 < -0.5)
      {
        tmp248 += 1.0;
        tmp247 -= 1.0;
      }
      if(fabs(tmp248) < 1e-10 && ((unsigned long)tmp247 & 1))
      {
        tmp244 = -pow(-tmp242, tmp245)*pow(tmp242, tmp246);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp242, tmp243);
      }
    }
  }
  else
  {
    tmp244 = pow(tmp242, tmp243);
  }
  if(isnan(tmp244) || isinf(tmp244))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp242, tmp243);
  }tmp249 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp244,"(r_init[476] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp249 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[476] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp249);
    }
  }
  (data->simulationInfo->realParameter[980] /* omega_c[476] PARAM */) = sqrt(tmp249);
  TRACE_POP
}

/*
equation index: 13051
type: SIMPLE_ASSIGN
r_init[475] = r_min + 475.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13051};
  (data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (475.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13052
type: SIMPLE_ASSIGN
omega_c[475] = sqrt(G * Md / (r_init[475] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13052(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13052};
  modelica_real tmp250;
  modelica_real tmp251;
  modelica_real tmp252;
  modelica_real tmp253;
  modelica_real tmp254;
  modelica_real tmp255;
  modelica_real tmp256;
  modelica_real tmp257;
  modelica_real tmp258;
  modelica_real tmp259;
  tmp250 = (data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */);
  tmp251 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp252 = (tmp250 * tmp250) + (tmp251 * tmp251);
  tmp253 = 1.5;
  if(tmp252 < 0.0 && tmp253 != 0.0)
  {
    tmp255 = modf(tmp253, &tmp256);
    
    if(tmp255 > 0.5)
    {
      tmp255 -= 1.0;
      tmp256 += 1.0;
    }
    else if(tmp255 < -0.5)
    {
      tmp255 += 1.0;
      tmp256 -= 1.0;
    }
    
    if(fabs(tmp255) < 1e-10)
      tmp254 = pow(tmp252, tmp256);
    else
    {
      tmp258 = modf(1.0/tmp253, &tmp257);
      if(tmp258 > 0.5)
      {
        tmp258 -= 1.0;
        tmp257 += 1.0;
      }
      else if(tmp258 < -0.5)
      {
        tmp258 += 1.0;
        tmp257 -= 1.0;
      }
      if(fabs(tmp258) < 1e-10 && ((unsigned long)tmp257 & 1))
      {
        tmp254 = -pow(-tmp252, tmp255)*pow(tmp252, tmp256);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp252, tmp253);
      }
    }
  }
  else
  {
    tmp254 = pow(tmp252, tmp253);
  }
  if(isnan(tmp254) || isinf(tmp254))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp252, tmp253);
  }tmp259 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp254,"(r_init[475] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp259 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[475] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp259);
    }
  }
  (data->simulationInfo->realParameter[979] /* omega_c[475] PARAM */) = sqrt(tmp259);
  TRACE_POP
}

/*
equation index: 13053
type: SIMPLE_ASSIGN
r_init[474] = r_min + 474.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13053};
  (data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (474.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13054
type: SIMPLE_ASSIGN
omega_c[474] = sqrt(G * Md / (r_init[474] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13054(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13054};
  modelica_real tmp260;
  modelica_real tmp261;
  modelica_real tmp262;
  modelica_real tmp263;
  modelica_real tmp264;
  modelica_real tmp265;
  modelica_real tmp266;
  modelica_real tmp267;
  modelica_real tmp268;
  modelica_real tmp269;
  tmp260 = (data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */);
  tmp261 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp262 = (tmp260 * tmp260) + (tmp261 * tmp261);
  tmp263 = 1.5;
  if(tmp262 < 0.0 && tmp263 != 0.0)
  {
    tmp265 = modf(tmp263, &tmp266);
    
    if(tmp265 > 0.5)
    {
      tmp265 -= 1.0;
      tmp266 += 1.0;
    }
    else if(tmp265 < -0.5)
    {
      tmp265 += 1.0;
      tmp266 -= 1.0;
    }
    
    if(fabs(tmp265) < 1e-10)
      tmp264 = pow(tmp262, tmp266);
    else
    {
      tmp268 = modf(1.0/tmp263, &tmp267);
      if(tmp268 > 0.5)
      {
        tmp268 -= 1.0;
        tmp267 += 1.0;
      }
      else if(tmp268 < -0.5)
      {
        tmp268 += 1.0;
        tmp267 -= 1.0;
      }
      if(fabs(tmp268) < 1e-10 && ((unsigned long)tmp267 & 1))
      {
        tmp264 = -pow(-tmp262, tmp265)*pow(tmp262, tmp266);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp262, tmp263);
      }
    }
  }
  else
  {
    tmp264 = pow(tmp262, tmp263);
  }
  if(isnan(tmp264) || isinf(tmp264))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp262, tmp263);
  }tmp269 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp264,"(r_init[474] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp269 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[474] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp269);
    }
  }
  (data->simulationInfo->realParameter[978] /* omega_c[474] PARAM */) = sqrt(tmp269);
  TRACE_POP
}

/*
equation index: 13055
type: SIMPLE_ASSIGN
r_init[473] = r_min + 473.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13055};
  (data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (473.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13056
type: SIMPLE_ASSIGN
omega_c[473] = sqrt(G * Md / (r_init[473] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13056(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13056};
  modelica_real tmp270;
  modelica_real tmp271;
  modelica_real tmp272;
  modelica_real tmp273;
  modelica_real tmp274;
  modelica_real tmp275;
  modelica_real tmp276;
  modelica_real tmp277;
  modelica_real tmp278;
  modelica_real tmp279;
  tmp270 = (data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */);
  tmp271 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp272 = (tmp270 * tmp270) + (tmp271 * tmp271);
  tmp273 = 1.5;
  if(tmp272 < 0.0 && tmp273 != 0.0)
  {
    tmp275 = modf(tmp273, &tmp276);
    
    if(tmp275 > 0.5)
    {
      tmp275 -= 1.0;
      tmp276 += 1.0;
    }
    else if(tmp275 < -0.5)
    {
      tmp275 += 1.0;
      tmp276 -= 1.0;
    }
    
    if(fabs(tmp275) < 1e-10)
      tmp274 = pow(tmp272, tmp276);
    else
    {
      tmp278 = modf(1.0/tmp273, &tmp277);
      if(tmp278 > 0.5)
      {
        tmp278 -= 1.0;
        tmp277 += 1.0;
      }
      else if(tmp278 < -0.5)
      {
        tmp278 += 1.0;
        tmp277 -= 1.0;
      }
      if(fabs(tmp278) < 1e-10 && ((unsigned long)tmp277 & 1))
      {
        tmp274 = -pow(-tmp272, tmp275)*pow(tmp272, tmp276);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp272, tmp273);
      }
    }
  }
  else
  {
    tmp274 = pow(tmp272, tmp273);
  }
  if(isnan(tmp274) || isinf(tmp274))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp272, tmp273);
  }tmp279 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp274,"(r_init[473] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp279 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[473] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp279);
    }
  }
  (data->simulationInfo->realParameter[977] /* omega_c[473] PARAM */) = sqrt(tmp279);
  TRACE_POP
}

/*
equation index: 13057
type: SIMPLE_ASSIGN
r_init[472] = r_min + 472.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13057};
  (data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (472.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13058
type: SIMPLE_ASSIGN
omega_c[472] = sqrt(G * Md / (r_init[472] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13058(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13058};
  modelica_real tmp280;
  modelica_real tmp281;
  modelica_real tmp282;
  modelica_real tmp283;
  modelica_real tmp284;
  modelica_real tmp285;
  modelica_real tmp286;
  modelica_real tmp287;
  modelica_real tmp288;
  modelica_real tmp289;
  tmp280 = (data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */);
  tmp281 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp282 = (tmp280 * tmp280) + (tmp281 * tmp281);
  tmp283 = 1.5;
  if(tmp282 < 0.0 && tmp283 != 0.0)
  {
    tmp285 = modf(tmp283, &tmp286);
    
    if(tmp285 > 0.5)
    {
      tmp285 -= 1.0;
      tmp286 += 1.0;
    }
    else if(tmp285 < -0.5)
    {
      tmp285 += 1.0;
      tmp286 -= 1.0;
    }
    
    if(fabs(tmp285) < 1e-10)
      tmp284 = pow(tmp282, tmp286);
    else
    {
      tmp288 = modf(1.0/tmp283, &tmp287);
      if(tmp288 > 0.5)
      {
        tmp288 -= 1.0;
        tmp287 += 1.0;
      }
      else if(tmp288 < -0.5)
      {
        tmp288 += 1.0;
        tmp287 -= 1.0;
      }
      if(fabs(tmp288) < 1e-10 && ((unsigned long)tmp287 & 1))
      {
        tmp284 = -pow(-tmp282, tmp285)*pow(tmp282, tmp286);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp282, tmp283);
      }
    }
  }
  else
  {
    tmp284 = pow(tmp282, tmp283);
  }
  if(isnan(tmp284) || isinf(tmp284))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp282, tmp283);
  }tmp289 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp284,"(r_init[472] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp289 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[472] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp289);
    }
  }
  (data->simulationInfo->realParameter[976] /* omega_c[472] PARAM */) = sqrt(tmp289);
  TRACE_POP
}

/*
equation index: 13059
type: SIMPLE_ASSIGN
r_init[471] = r_min + 471.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13059(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13059};
  (data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (471.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13060
type: SIMPLE_ASSIGN
omega_c[471] = sqrt(G * Md / (r_init[471] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13060};
  modelica_real tmp290;
  modelica_real tmp291;
  modelica_real tmp292;
  modelica_real tmp293;
  modelica_real tmp294;
  modelica_real tmp295;
  modelica_real tmp296;
  modelica_real tmp297;
  modelica_real tmp298;
  modelica_real tmp299;
  tmp290 = (data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */);
  tmp291 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp292 = (tmp290 * tmp290) + (tmp291 * tmp291);
  tmp293 = 1.5;
  if(tmp292 < 0.0 && tmp293 != 0.0)
  {
    tmp295 = modf(tmp293, &tmp296);
    
    if(tmp295 > 0.5)
    {
      tmp295 -= 1.0;
      tmp296 += 1.0;
    }
    else if(tmp295 < -0.5)
    {
      tmp295 += 1.0;
      tmp296 -= 1.0;
    }
    
    if(fabs(tmp295) < 1e-10)
      tmp294 = pow(tmp292, tmp296);
    else
    {
      tmp298 = modf(1.0/tmp293, &tmp297);
      if(tmp298 > 0.5)
      {
        tmp298 -= 1.0;
        tmp297 += 1.0;
      }
      else if(tmp298 < -0.5)
      {
        tmp298 += 1.0;
        tmp297 -= 1.0;
      }
      if(fabs(tmp298) < 1e-10 && ((unsigned long)tmp297 & 1))
      {
        tmp294 = -pow(-tmp292, tmp295)*pow(tmp292, tmp296);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp292, tmp293);
      }
    }
  }
  else
  {
    tmp294 = pow(tmp292, tmp293);
  }
  if(isnan(tmp294) || isinf(tmp294))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp292, tmp293);
  }tmp299 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp294,"(r_init[471] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp299 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[471] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp299);
    }
  }
  (data->simulationInfo->realParameter[975] /* omega_c[471] PARAM */) = sqrt(tmp299);
  TRACE_POP
}

/*
equation index: 13061
type: SIMPLE_ASSIGN
r_init[470] = r_min + 470.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13061};
  (data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (470.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13062
type: SIMPLE_ASSIGN
omega_c[470] = sqrt(G * Md / (r_init[470] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13062(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13062};
  modelica_real tmp300;
  modelica_real tmp301;
  modelica_real tmp302;
  modelica_real tmp303;
  modelica_real tmp304;
  modelica_real tmp305;
  modelica_real tmp306;
  modelica_real tmp307;
  modelica_real tmp308;
  modelica_real tmp309;
  tmp300 = (data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */);
  tmp301 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp302 = (tmp300 * tmp300) + (tmp301 * tmp301);
  tmp303 = 1.5;
  if(tmp302 < 0.0 && tmp303 != 0.0)
  {
    tmp305 = modf(tmp303, &tmp306);
    
    if(tmp305 > 0.5)
    {
      tmp305 -= 1.0;
      tmp306 += 1.0;
    }
    else if(tmp305 < -0.5)
    {
      tmp305 += 1.0;
      tmp306 -= 1.0;
    }
    
    if(fabs(tmp305) < 1e-10)
      tmp304 = pow(tmp302, tmp306);
    else
    {
      tmp308 = modf(1.0/tmp303, &tmp307);
      if(tmp308 > 0.5)
      {
        tmp308 -= 1.0;
        tmp307 += 1.0;
      }
      else if(tmp308 < -0.5)
      {
        tmp308 += 1.0;
        tmp307 -= 1.0;
      }
      if(fabs(tmp308) < 1e-10 && ((unsigned long)tmp307 & 1))
      {
        tmp304 = -pow(-tmp302, tmp305)*pow(tmp302, tmp306);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp302, tmp303);
      }
    }
  }
  else
  {
    tmp304 = pow(tmp302, tmp303);
  }
  if(isnan(tmp304) || isinf(tmp304))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp302, tmp303);
  }tmp309 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp304,"(r_init[470] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp309 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[470] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp309);
    }
  }
  (data->simulationInfo->realParameter[974] /* omega_c[470] PARAM */) = sqrt(tmp309);
  TRACE_POP
}

/*
equation index: 13063
type: SIMPLE_ASSIGN
r_init[469] = r_min + 469.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13063};
  (data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (469.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13064
type: SIMPLE_ASSIGN
omega_c[469] = sqrt(G * Md / (r_init[469] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13064(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13064};
  modelica_real tmp310;
  modelica_real tmp311;
  modelica_real tmp312;
  modelica_real tmp313;
  modelica_real tmp314;
  modelica_real tmp315;
  modelica_real tmp316;
  modelica_real tmp317;
  modelica_real tmp318;
  modelica_real tmp319;
  tmp310 = (data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */);
  tmp311 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp312 = (tmp310 * tmp310) + (tmp311 * tmp311);
  tmp313 = 1.5;
  if(tmp312 < 0.0 && tmp313 != 0.0)
  {
    tmp315 = modf(tmp313, &tmp316);
    
    if(tmp315 > 0.5)
    {
      tmp315 -= 1.0;
      tmp316 += 1.0;
    }
    else if(tmp315 < -0.5)
    {
      tmp315 += 1.0;
      tmp316 -= 1.0;
    }
    
    if(fabs(tmp315) < 1e-10)
      tmp314 = pow(tmp312, tmp316);
    else
    {
      tmp318 = modf(1.0/tmp313, &tmp317);
      if(tmp318 > 0.5)
      {
        tmp318 -= 1.0;
        tmp317 += 1.0;
      }
      else if(tmp318 < -0.5)
      {
        tmp318 += 1.0;
        tmp317 -= 1.0;
      }
      if(fabs(tmp318) < 1e-10 && ((unsigned long)tmp317 & 1))
      {
        tmp314 = -pow(-tmp312, tmp315)*pow(tmp312, tmp316);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp312, tmp313);
      }
    }
  }
  else
  {
    tmp314 = pow(tmp312, tmp313);
  }
  if(isnan(tmp314) || isinf(tmp314))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp312, tmp313);
  }tmp319 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp314,"(r_init[469] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp319 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[469] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp319);
    }
  }
  (data->simulationInfo->realParameter[973] /* omega_c[469] PARAM */) = sqrt(tmp319);
  TRACE_POP
}

/*
equation index: 13065
type: SIMPLE_ASSIGN
r_init[468] = r_min + 468.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13065(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13065};
  (data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (468.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13066
type: SIMPLE_ASSIGN
omega_c[468] = sqrt(G * Md / (r_init[468] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13066(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13066};
  modelica_real tmp320;
  modelica_real tmp321;
  modelica_real tmp322;
  modelica_real tmp323;
  modelica_real tmp324;
  modelica_real tmp325;
  modelica_real tmp326;
  modelica_real tmp327;
  modelica_real tmp328;
  modelica_real tmp329;
  tmp320 = (data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */);
  tmp321 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp322 = (tmp320 * tmp320) + (tmp321 * tmp321);
  tmp323 = 1.5;
  if(tmp322 < 0.0 && tmp323 != 0.0)
  {
    tmp325 = modf(tmp323, &tmp326);
    
    if(tmp325 > 0.5)
    {
      tmp325 -= 1.0;
      tmp326 += 1.0;
    }
    else if(tmp325 < -0.5)
    {
      tmp325 += 1.0;
      tmp326 -= 1.0;
    }
    
    if(fabs(tmp325) < 1e-10)
      tmp324 = pow(tmp322, tmp326);
    else
    {
      tmp328 = modf(1.0/tmp323, &tmp327);
      if(tmp328 > 0.5)
      {
        tmp328 -= 1.0;
        tmp327 += 1.0;
      }
      else if(tmp328 < -0.5)
      {
        tmp328 += 1.0;
        tmp327 -= 1.0;
      }
      if(fabs(tmp328) < 1e-10 && ((unsigned long)tmp327 & 1))
      {
        tmp324 = -pow(-tmp322, tmp325)*pow(tmp322, tmp326);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp322, tmp323);
      }
    }
  }
  else
  {
    tmp324 = pow(tmp322, tmp323);
  }
  if(isnan(tmp324) || isinf(tmp324))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp322, tmp323);
  }tmp329 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp324,"(r_init[468] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp329 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[468] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp329);
    }
  }
  (data->simulationInfo->realParameter[972] /* omega_c[468] PARAM */) = sqrt(tmp329);
  TRACE_POP
}

/*
equation index: 13067
type: SIMPLE_ASSIGN
r_init[467] = r_min + 467.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13067};
  (data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (467.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13068
type: SIMPLE_ASSIGN
omega_c[467] = sqrt(G * Md / (r_init[467] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13068(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13068};
  modelica_real tmp330;
  modelica_real tmp331;
  modelica_real tmp332;
  modelica_real tmp333;
  modelica_real tmp334;
  modelica_real tmp335;
  modelica_real tmp336;
  modelica_real tmp337;
  modelica_real tmp338;
  modelica_real tmp339;
  tmp330 = (data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */);
  tmp331 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp332 = (tmp330 * tmp330) + (tmp331 * tmp331);
  tmp333 = 1.5;
  if(tmp332 < 0.0 && tmp333 != 0.0)
  {
    tmp335 = modf(tmp333, &tmp336);
    
    if(tmp335 > 0.5)
    {
      tmp335 -= 1.0;
      tmp336 += 1.0;
    }
    else if(tmp335 < -0.5)
    {
      tmp335 += 1.0;
      tmp336 -= 1.0;
    }
    
    if(fabs(tmp335) < 1e-10)
      tmp334 = pow(tmp332, tmp336);
    else
    {
      tmp338 = modf(1.0/tmp333, &tmp337);
      if(tmp338 > 0.5)
      {
        tmp338 -= 1.0;
        tmp337 += 1.0;
      }
      else if(tmp338 < -0.5)
      {
        tmp338 += 1.0;
        tmp337 -= 1.0;
      }
      if(fabs(tmp338) < 1e-10 && ((unsigned long)tmp337 & 1))
      {
        tmp334 = -pow(-tmp332, tmp335)*pow(tmp332, tmp336);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp332, tmp333);
      }
    }
  }
  else
  {
    tmp334 = pow(tmp332, tmp333);
  }
  if(isnan(tmp334) || isinf(tmp334))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp332, tmp333);
  }tmp339 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp334,"(r_init[467] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp339 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[467] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp339);
    }
  }
  (data->simulationInfo->realParameter[971] /* omega_c[467] PARAM */) = sqrt(tmp339);
  TRACE_POP
}

/*
equation index: 13069
type: SIMPLE_ASSIGN
r_init[466] = r_min + 466.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13069};
  (data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (466.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13070
type: SIMPLE_ASSIGN
omega_c[466] = sqrt(G * Md / (r_init[466] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13070(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13070};
  modelica_real tmp340;
  modelica_real tmp341;
  modelica_real tmp342;
  modelica_real tmp343;
  modelica_real tmp344;
  modelica_real tmp345;
  modelica_real tmp346;
  modelica_real tmp347;
  modelica_real tmp348;
  modelica_real tmp349;
  tmp340 = (data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */);
  tmp341 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp342 = (tmp340 * tmp340) + (tmp341 * tmp341);
  tmp343 = 1.5;
  if(tmp342 < 0.0 && tmp343 != 0.0)
  {
    tmp345 = modf(tmp343, &tmp346);
    
    if(tmp345 > 0.5)
    {
      tmp345 -= 1.0;
      tmp346 += 1.0;
    }
    else if(tmp345 < -0.5)
    {
      tmp345 += 1.0;
      tmp346 -= 1.0;
    }
    
    if(fabs(tmp345) < 1e-10)
      tmp344 = pow(tmp342, tmp346);
    else
    {
      tmp348 = modf(1.0/tmp343, &tmp347);
      if(tmp348 > 0.5)
      {
        tmp348 -= 1.0;
        tmp347 += 1.0;
      }
      else if(tmp348 < -0.5)
      {
        tmp348 += 1.0;
        tmp347 -= 1.0;
      }
      if(fabs(tmp348) < 1e-10 && ((unsigned long)tmp347 & 1))
      {
        tmp344 = -pow(-tmp342, tmp345)*pow(tmp342, tmp346);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp342, tmp343);
      }
    }
  }
  else
  {
    tmp344 = pow(tmp342, tmp343);
  }
  if(isnan(tmp344) || isinf(tmp344))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp342, tmp343);
  }tmp349 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp344,"(r_init[466] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp349 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[466] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp349);
    }
  }
  (data->simulationInfo->realParameter[970] /* omega_c[466] PARAM */) = sqrt(tmp349);
  TRACE_POP
}

/*
equation index: 13071
type: SIMPLE_ASSIGN
r_init[465] = r_min + 465.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13071};
  (data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (465.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13072
type: SIMPLE_ASSIGN
omega_c[465] = sqrt(G * Md / (r_init[465] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13072(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13072};
  modelica_real tmp350;
  modelica_real tmp351;
  modelica_real tmp352;
  modelica_real tmp353;
  modelica_real tmp354;
  modelica_real tmp355;
  modelica_real tmp356;
  modelica_real tmp357;
  modelica_real tmp358;
  modelica_real tmp359;
  tmp350 = (data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */);
  tmp351 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp352 = (tmp350 * tmp350) + (tmp351 * tmp351);
  tmp353 = 1.5;
  if(tmp352 < 0.0 && tmp353 != 0.0)
  {
    tmp355 = modf(tmp353, &tmp356);
    
    if(tmp355 > 0.5)
    {
      tmp355 -= 1.0;
      tmp356 += 1.0;
    }
    else if(tmp355 < -0.5)
    {
      tmp355 += 1.0;
      tmp356 -= 1.0;
    }
    
    if(fabs(tmp355) < 1e-10)
      tmp354 = pow(tmp352, tmp356);
    else
    {
      tmp358 = modf(1.0/tmp353, &tmp357);
      if(tmp358 > 0.5)
      {
        tmp358 -= 1.0;
        tmp357 += 1.0;
      }
      else if(tmp358 < -0.5)
      {
        tmp358 += 1.0;
        tmp357 -= 1.0;
      }
      if(fabs(tmp358) < 1e-10 && ((unsigned long)tmp357 & 1))
      {
        tmp354 = -pow(-tmp352, tmp355)*pow(tmp352, tmp356);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp352, tmp353);
      }
    }
  }
  else
  {
    tmp354 = pow(tmp352, tmp353);
  }
  if(isnan(tmp354) || isinf(tmp354))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp352, tmp353);
  }tmp359 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp354,"(r_init[465] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp359 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[465] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp359);
    }
  }
  (data->simulationInfo->realParameter[969] /* omega_c[465] PARAM */) = sqrt(tmp359);
  TRACE_POP
}

/*
equation index: 13073
type: SIMPLE_ASSIGN
r_init[464] = r_min + 464.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13073};
  (data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (464.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13074
type: SIMPLE_ASSIGN
omega_c[464] = sqrt(G * Md / (r_init[464] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13074(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13074};
  modelica_real tmp360;
  modelica_real tmp361;
  modelica_real tmp362;
  modelica_real tmp363;
  modelica_real tmp364;
  modelica_real tmp365;
  modelica_real tmp366;
  modelica_real tmp367;
  modelica_real tmp368;
  modelica_real tmp369;
  tmp360 = (data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */);
  tmp361 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp362 = (tmp360 * tmp360) + (tmp361 * tmp361);
  tmp363 = 1.5;
  if(tmp362 < 0.0 && tmp363 != 0.0)
  {
    tmp365 = modf(tmp363, &tmp366);
    
    if(tmp365 > 0.5)
    {
      tmp365 -= 1.0;
      tmp366 += 1.0;
    }
    else if(tmp365 < -0.5)
    {
      tmp365 += 1.0;
      tmp366 -= 1.0;
    }
    
    if(fabs(tmp365) < 1e-10)
      tmp364 = pow(tmp362, tmp366);
    else
    {
      tmp368 = modf(1.0/tmp363, &tmp367);
      if(tmp368 > 0.5)
      {
        tmp368 -= 1.0;
        tmp367 += 1.0;
      }
      else if(tmp368 < -0.5)
      {
        tmp368 += 1.0;
        tmp367 -= 1.0;
      }
      if(fabs(tmp368) < 1e-10 && ((unsigned long)tmp367 & 1))
      {
        tmp364 = -pow(-tmp362, tmp365)*pow(tmp362, tmp366);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp362, tmp363);
      }
    }
  }
  else
  {
    tmp364 = pow(tmp362, tmp363);
  }
  if(isnan(tmp364) || isinf(tmp364))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp362, tmp363);
  }tmp369 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp364,"(r_init[464] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp369 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[464] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp369);
    }
  }
  (data->simulationInfo->realParameter[968] /* omega_c[464] PARAM */) = sqrt(tmp369);
  TRACE_POP
}

/*
equation index: 13075
type: SIMPLE_ASSIGN
r_init[463] = r_min + 463.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13075(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13075};
  (data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (463.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13076
type: SIMPLE_ASSIGN
omega_c[463] = sqrt(G * Md / (r_init[463] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13076};
  modelica_real tmp370;
  modelica_real tmp371;
  modelica_real tmp372;
  modelica_real tmp373;
  modelica_real tmp374;
  modelica_real tmp375;
  modelica_real tmp376;
  modelica_real tmp377;
  modelica_real tmp378;
  modelica_real tmp379;
  tmp370 = (data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */);
  tmp371 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp372 = (tmp370 * tmp370) + (tmp371 * tmp371);
  tmp373 = 1.5;
  if(tmp372 < 0.0 && tmp373 != 0.0)
  {
    tmp375 = modf(tmp373, &tmp376);
    
    if(tmp375 > 0.5)
    {
      tmp375 -= 1.0;
      tmp376 += 1.0;
    }
    else if(tmp375 < -0.5)
    {
      tmp375 += 1.0;
      tmp376 -= 1.0;
    }
    
    if(fabs(tmp375) < 1e-10)
      tmp374 = pow(tmp372, tmp376);
    else
    {
      tmp378 = modf(1.0/tmp373, &tmp377);
      if(tmp378 > 0.5)
      {
        tmp378 -= 1.0;
        tmp377 += 1.0;
      }
      else if(tmp378 < -0.5)
      {
        tmp378 += 1.0;
        tmp377 -= 1.0;
      }
      if(fabs(tmp378) < 1e-10 && ((unsigned long)tmp377 & 1))
      {
        tmp374 = -pow(-tmp372, tmp375)*pow(tmp372, tmp376);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp372, tmp373);
      }
    }
  }
  else
  {
    tmp374 = pow(tmp372, tmp373);
  }
  if(isnan(tmp374) || isinf(tmp374))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp372, tmp373);
  }tmp379 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp374,"(r_init[463] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp379 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[463] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp379);
    }
  }
  (data->simulationInfo->realParameter[967] /* omega_c[463] PARAM */) = sqrt(tmp379);
  TRACE_POP
}

/*
equation index: 13077
type: SIMPLE_ASSIGN
r_init[462] = r_min + 462.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13077};
  (data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (462.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13078
type: SIMPLE_ASSIGN
omega_c[462] = sqrt(G * Md / (r_init[462] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13078(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13078};
  modelica_real tmp380;
  modelica_real tmp381;
  modelica_real tmp382;
  modelica_real tmp383;
  modelica_real tmp384;
  modelica_real tmp385;
  modelica_real tmp386;
  modelica_real tmp387;
  modelica_real tmp388;
  modelica_real tmp389;
  tmp380 = (data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */);
  tmp381 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp382 = (tmp380 * tmp380) + (tmp381 * tmp381);
  tmp383 = 1.5;
  if(tmp382 < 0.0 && tmp383 != 0.0)
  {
    tmp385 = modf(tmp383, &tmp386);
    
    if(tmp385 > 0.5)
    {
      tmp385 -= 1.0;
      tmp386 += 1.0;
    }
    else if(tmp385 < -0.5)
    {
      tmp385 += 1.0;
      tmp386 -= 1.0;
    }
    
    if(fabs(tmp385) < 1e-10)
      tmp384 = pow(tmp382, tmp386);
    else
    {
      tmp388 = modf(1.0/tmp383, &tmp387);
      if(tmp388 > 0.5)
      {
        tmp388 -= 1.0;
        tmp387 += 1.0;
      }
      else if(tmp388 < -0.5)
      {
        tmp388 += 1.0;
        tmp387 -= 1.0;
      }
      if(fabs(tmp388) < 1e-10 && ((unsigned long)tmp387 & 1))
      {
        tmp384 = -pow(-tmp382, tmp385)*pow(tmp382, tmp386);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp382, tmp383);
      }
    }
  }
  else
  {
    tmp384 = pow(tmp382, tmp383);
  }
  if(isnan(tmp384) || isinf(tmp384))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp382, tmp383);
  }tmp389 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp384,"(r_init[462] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp389 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[462] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp389);
    }
  }
  (data->simulationInfo->realParameter[966] /* omega_c[462] PARAM */) = sqrt(tmp389);
  TRACE_POP
}

/*
equation index: 13079
type: SIMPLE_ASSIGN
r_init[461] = r_min + 461.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13079};
  (data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (461.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13080
type: SIMPLE_ASSIGN
omega_c[461] = sqrt(G * Md / (r_init[461] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13080(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13080};
  modelica_real tmp390;
  modelica_real tmp391;
  modelica_real tmp392;
  modelica_real tmp393;
  modelica_real tmp394;
  modelica_real tmp395;
  modelica_real tmp396;
  modelica_real tmp397;
  modelica_real tmp398;
  modelica_real tmp399;
  tmp390 = (data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */);
  tmp391 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp392 = (tmp390 * tmp390) + (tmp391 * tmp391);
  tmp393 = 1.5;
  if(tmp392 < 0.0 && tmp393 != 0.0)
  {
    tmp395 = modf(tmp393, &tmp396);
    
    if(tmp395 > 0.5)
    {
      tmp395 -= 1.0;
      tmp396 += 1.0;
    }
    else if(tmp395 < -0.5)
    {
      tmp395 += 1.0;
      tmp396 -= 1.0;
    }
    
    if(fabs(tmp395) < 1e-10)
      tmp394 = pow(tmp392, tmp396);
    else
    {
      tmp398 = modf(1.0/tmp393, &tmp397);
      if(tmp398 > 0.5)
      {
        tmp398 -= 1.0;
        tmp397 += 1.0;
      }
      else if(tmp398 < -0.5)
      {
        tmp398 += 1.0;
        tmp397 -= 1.0;
      }
      if(fabs(tmp398) < 1e-10 && ((unsigned long)tmp397 & 1))
      {
        tmp394 = -pow(-tmp392, tmp395)*pow(tmp392, tmp396);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp392, tmp393);
      }
    }
  }
  else
  {
    tmp394 = pow(tmp392, tmp393);
  }
  if(isnan(tmp394) || isinf(tmp394))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp392, tmp393);
  }tmp399 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp394,"(r_init[461] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp399 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[461] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp399);
    }
  }
  (data->simulationInfo->realParameter[965] /* omega_c[461] PARAM */) = sqrt(tmp399);
  TRACE_POP
}

/*
equation index: 13081
type: SIMPLE_ASSIGN
r_init[460] = r_min + 460.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13081(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13081};
  (data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (460.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13082
type: SIMPLE_ASSIGN
omega_c[460] = sqrt(G * Md / (r_init[460] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13082(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13082};
  modelica_real tmp400;
  modelica_real tmp401;
  modelica_real tmp402;
  modelica_real tmp403;
  modelica_real tmp404;
  modelica_real tmp405;
  modelica_real tmp406;
  modelica_real tmp407;
  modelica_real tmp408;
  modelica_real tmp409;
  tmp400 = (data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */);
  tmp401 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp402 = (tmp400 * tmp400) + (tmp401 * tmp401);
  tmp403 = 1.5;
  if(tmp402 < 0.0 && tmp403 != 0.0)
  {
    tmp405 = modf(tmp403, &tmp406);
    
    if(tmp405 > 0.5)
    {
      tmp405 -= 1.0;
      tmp406 += 1.0;
    }
    else if(tmp405 < -0.5)
    {
      tmp405 += 1.0;
      tmp406 -= 1.0;
    }
    
    if(fabs(tmp405) < 1e-10)
      tmp404 = pow(tmp402, tmp406);
    else
    {
      tmp408 = modf(1.0/tmp403, &tmp407);
      if(tmp408 > 0.5)
      {
        tmp408 -= 1.0;
        tmp407 += 1.0;
      }
      else if(tmp408 < -0.5)
      {
        tmp408 += 1.0;
        tmp407 -= 1.0;
      }
      if(fabs(tmp408) < 1e-10 && ((unsigned long)tmp407 & 1))
      {
        tmp404 = -pow(-tmp402, tmp405)*pow(tmp402, tmp406);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp402, tmp403);
      }
    }
  }
  else
  {
    tmp404 = pow(tmp402, tmp403);
  }
  if(isnan(tmp404) || isinf(tmp404))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp402, tmp403);
  }tmp409 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp404,"(r_init[460] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp409 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[460] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp409);
    }
  }
  (data->simulationInfo->realParameter[964] /* omega_c[460] PARAM */) = sqrt(tmp409);
  TRACE_POP
}

/*
equation index: 13083
type: SIMPLE_ASSIGN
r_init[459] = r_min + 459.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13083};
  (data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (459.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13084
type: SIMPLE_ASSIGN
omega_c[459] = sqrt(G * Md / (r_init[459] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13084(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13084};
  modelica_real tmp410;
  modelica_real tmp411;
  modelica_real tmp412;
  modelica_real tmp413;
  modelica_real tmp414;
  modelica_real tmp415;
  modelica_real tmp416;
  modelica_real tmp417;
  modelica_real tmp418;
  modelica_real tmp419;
  tmp410 = (data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */);
  tmp411 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp412 = (tmp410 * tmp410) + (tmp411 * tmp411);
  tmp413 = 1.5;
  if(tmp412 < 0.0 && tmp413 != 0.0)
  {
    tmp415 = modf(tmp413, &tmp416);
    
    if(tmp415 > 0.5)
    {
      tmp415 -= 1.0;
      tmp416 += 1.0;
    }
    else if(tmp415 < -0.5)
    {
      tmp415 += 1.0;
      tmp416 -= 1.0;
    }
    
    if(fabs(tmp415) < 1e-10)
      tmp414 = pow(tmp412, tmp416);
    else
    {
      tmp418 = modf(1.0/tmp413, &tmp417);
      if(tmp418 > 0.5)
      {
        tmp418 -= 1.0;
        tmp417 += 1.0;
      }
      else if(tmp418 < -0.5)
      {
        tmp418 += 1.0;
        tmp417 -= 1.0;
      }
      if(fabs(tmp418) < 1e-10 && ((unsigned long)tmp417 & 1))
      {
        tmp414 = -pow(-tmp412, tmp415)*pow(tmp412, tmp416);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp412, tmp413);
      }
    }
  }
  else
  {
    tmp414 = pow(tmp412, tmp413);
  }
  if(isnan(tmp414) || isinf(tmp414))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp412, tmp413);
  }tmp419 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp414,"(r_init[459] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp419 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[459] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp419);
    }
  }
  (data->simulationInfo->realParameter[963] /* omega_c[459] PARAM */) = sqrt(tmp419);
  TRACE_POP
}

/*
equation index: 13085
type: SIMPLE_ASSIGN
r_init[458] = r_min + 458.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13085};
  (data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (458.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13086
type: SIMPLE_ASSIGN
omega_c[458] = sqrt(G * Md / (r_init[458] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13086(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13086};
  modelica_real tmp420;
  modelica_real tmp421;
  modelica_real tmp422;
  modelica_real tmp423;
  modelica_real tmp424;
  modelica_real tmp425;
  modelica_real tmp426;
  modelica_real tmp427;
  modelica_real tmp428;
  modelica_real tmp429;
  tmp420 = (data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */);
  tmp421 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp422 = (tmp420 * tmp420) + (tmp421 * tmp421);
  tmp423 = 1.5;
  if(tmp422 < 0.0 && tmp423 != 0.0)
  {
    tmp425 = modf(tmp423, &tmp426);
    
    if(tmp425 > 0.5)
    {
      tmp425 -= 1.0;
      tmp426 += 1.0;
    }
    else if(tmp425 < -0.5)
    {
      tmp425 += 1.0;
      tmp426 -= 1.0;
    }
    
    if(fabs(tmp425) < 1e-10)
      tmp424 = pow(tmp422, tmp426);
    else
    {
      tmp428 = modf(1.0/tmp423, &tmp427);
      if(tmp428 > 0.5)
      {
        tmp428 -= 1.0;
        tmp427 += 1.0;
      }
      else if(tmp428 < -0.5)
      {
        tmp428 += 1.0;
        tmp427 -= 1.0;
      }
      if(fabs(tmp428) < 1e-10 && ((unsigned long)tmp427 & 1))
      {
        tmp424 = -pow(-tmp422, tmp425)*pow(tmp422, tmp426);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp422, tmp423);
      }
    }
  }
  else
  {
    tmp424 = pow(tmp422, tmp423);
  }
  if(isnan(tmp424) || isinf(tmp424))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp422, tmp423);
  }tmp429 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp424,"(r_init[458] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp429 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[458] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp429);
    }
  }
  (data->simulationInfo->realParameter[962] /* omega_c[458] PARAM */) = sqrt(tmp429);
  TRACE_POP
}

/*
equation index: 13087
type: SIMPLE_ASSIGN
r_init[457] = r_min + 457.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13087};
  (data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (457.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13088
type: SIMPLE_ASSIGN
omega_c[457] = sqrt(G * Md / (r_init[457] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13088(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13088};
  modelica_real tmp430;
  modelica_real tmp431;
  modelica_real tmp432;
  modelica_real tmp433;
  modelica_real tmp434;
  modelica_real tmp435;
  modelica_real tmp436;
  modelica_real tmp437;
  modelica_real tmp438;
  modelica_real tmp439;
  tmp430 = (data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */);
  tmp431 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp432 = (tmp430 * tmp430) + (tmp431 * tmp431);
  tmp433 = 1.5;
  if(tmp432 < 0.0 && tmp433 != 0.0)
  {
    tmp435 = modf(tmp433, &tmp436);
    
    if(tmp435 > 0.5)
    {
      tmp435 -= 1.0;
      tmp436 += 1.0;
    }
    else if(tmp435 < -0.5)
    {
      tmp435 += 1.0;
      tmp436 -= 1.0;
    }
    
    if(fabs(tmp435) < 1e-10)
      tmp434 = pow(tmp432, tmp436);
    else
    {
      tmp438 = modf(1.0/tmp433, &tmp437);
      if(tmp438 > 0.5)
      {
        tmp438 -= 1.0;
        tmp437 += 1.0;
      }
      else if(tmp438 < -0.5)
      {
        tmp438 += 1.0;
        tmp437 -= 1.0;
      }
      if(fabs(tmp438) < 1e-10 && ((unsigned long)tmp437 & 1))
      {
        tmp434 = -pow(-tmp432, tmp435)*pow(tmp432, tmp436);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp432, tmp433);
      }
    }
  }
  else
  {
    tmp434 = pow(tmp432, tmp433);
  }
  if(isnan(tmp434) || isinf(tmp434))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp432, tmp433);
  }tmp439 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp434,"(r_init[457] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp439 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[457] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp439);
    }
  }
  (data->simulationInfo->realParameter[961] /* omega_c[457] PARAM */) = sqrt(tmp439);
  TRACE_POP
}

/*
equation index: 13089
type: SIMPLE_ASSIGN
r_init[456] = r_min + 456.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13089};
  (data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (456.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13090
type: SIMPLE_ASSIGN
omega_c[456] = sqrt(G * Md / (r_init[456] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13090(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13090};
  modelica_real tmp440;
  modelica_real tmp441;
  modelica_real tmp442;
  modelica_real tmp443;
  modelica_real tmp444;
  modelica_real tmp445;
  modelica_real tmp446;
  modelica_real tmp447;
  modelica_real tmp448;
  modelica_real tmp449;
  tmp440 = (data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */);
  tmp441 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp442 = (tmp440 * tmp440) + (tmp441 * tmp441);
  tmp443 = 1.5;
  if(tmp442 < 0.0 && tmp443 != 0.0)
  {
    tmp445 = modf(tmp443, &tmp446);
    
    if(tmp445 > 0.5)
    {
      tmp445 -= 1.0;
      tmp446 += 1.0;
    }
    else if(tmp445 < -0.5)
    {
      tmp445 += 1.0;
      tmp446 -= 1.0;
    }
    
    if(fabs(tmp445) < 1e-10)
      tmp444 = pow(tmp442, tmp446);
    else
    {
      tmp448 = modf(1.0/tmp443, &tmp447);
      if(tmp448 > 0.5)
      {
        tmp448 -= 1.0;
        tmp447 += 1.0;
      }
      else if(tmp448 < -0.5)
      {
        tmp448 += 1.0;
        tmp447 -= 1.0;
      }
      if(fabs(tmp448) < 1e-10 && ((unsigned long)tmp447 & 1))
      {
        tmp444 = -pow(-tmp442, tmp445)*pow(tmp442, tmp446);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp442, tmp443);
      }
    }
  }
  else
  {
    tmp444 = pow(tmp442, tmp443);
  }
  if(isnan(tmp444) || isinf(tmp444))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp442, tmp443);
  }tmp449 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp444,"(r_init[456] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp449 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[456] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp449);
    }
  }
  (data->simulationInfo->realParameter[960] /* omega_c[456] PARAM */) = sqrt(tmp449);
  TRACE_POP
}

/*
equation index: 13091
type: SIMPLE_ASSIGN
r_init[455] = r_min + 455.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13091(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13091};
  (data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (455.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13092
type: SIMPLE_ASSIGN
omega_c[455] = sqrt(G * Md / (r_init[455] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13092};
  modelica_real tmp450;
  modelica_real tmp451;
  modelica_real tmp452;
  modelica_real tmp453;
  modelica_real tmp454;
  modelica_real tmp455;
  modelica_real tmp456;
  modelica_real tmp457;
  modelica_real tmp458;
  modelica_real tmp459;
  tmp450 = (data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */);
  tmp451 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp452 = (tmp450 * tmp450) + (tmp451 * tmp451);
  tmp453 = 1.5;
  if(tmp452 < 0.0 && tmp453 != 0.0)
  {
    tmp455 = modf(tmp453, &tmp456);
    
    if(tmp455 > 0.5)
    {
      tmp455 -= 1.0;
      tmp456 += 1.0;
    }
    else if(tmp455 < -0.5)
    {
      tmp455 += 1.0;
      tmp456 -= 1.0;
    }
    
    if(fabs(tmp455) < 1e-10)
      tmp454 = pow(tmp452, tmp456);
    else
    {
      tmp458 = modf(1.0/tmp453, &tmp457);
      if(tmp458 > 0.5)
      {
        tmp458 -= 1.0;
        tmp457 += 1.0;
      }
      else if(tmp458 < -0.5)
      {
        tmp458 += 1.0;
        tmp457 -= 1.0;
      }
      if(fabs(tmp458) < 1e-10 && ((unsigned long)tmp457 & 1))
      {
        tmp454 = -pow(-tmp452, tmp455)*pow(tmp452, tmp456);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp452, tmp453);
      }
    }
  }
  else
  {
    tmp454 = pow(tmp452, tmp453);
  }
  if(isnan(tmp454) || isinf(tmp454))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp452, tmp453);
  }tmp459 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp454,"(r_init[455] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp459 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[455] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp459);
    }
  }
  (data->simulationInfo->realParameter[959] /* omega_c[455] PARAM */) = sqrt(tmp459);
  TRACE_POP
}

/*
equation index: 13093
type: SIMPLE_ASSIGN
r_init[454] = r_min + 454.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13093};
  (data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (454.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13094
type: SIMPLE_ASSIGN
omega_c[454] = sqrt(G * Md / (r_init[454] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13094(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13094};
  modelica_real tmp460;
  modelica_real tmp461;
  modelica_real tmp462;
  modelica_real tmp463;
  modelica_real tmp464;
  modelica_real tmp465;
  modelica_real tmp466;
  modelica_real tmp467;
  modelica_real tmp468;
  modelica_real tmp469;
  tmp460 = (data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */);
  tmp461 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp462 = (tmp460 * tmp460) + (tmp461 * tmp461);
  tmp463 = 1.5;
  if(tmp462 < 0.0 && tmp463 != 0.0)
  {
    tmp465 = modf(tmp463, &tmp466);
    
    if(tmp465 > 0.5)
    {
      tmp465 -= 1.0;
      tmp466 += 1.0;
    }
    else if(tmp465 < -0.5)
    {
      tmp465 += 1.0;
      tmp466 -= 1.0;
    }
    
    if(fabs(tmp465) < 1e-10)
      tmp464 = pow(tmp462, tmp466);
    else
    {
      tmp468 = modf(1.0/tmp463, &tmp467);
      if(tmp468 > 0.5)
      {
        tmp468 -= 1.0;
        tmp467 += 1.0;
      }
      else if(tmp468 < -0.5)
      {
        tmp468 += 1.0;
        tmp467 -= 1.0;
      }
      if(fabs(tmp468) < 1e-10 && ((unsigned long)tmp467 & 1))
      {
        tmp464 = -pow(-tmp462, tmp465)*pow(tmp462, tmp466);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp462, tmp463);
      }
    }
  }
  else
  {
    tmp464 = pow(tmp462, tmp463);
  }
  if(isnan(tmp464) || isinf(tmp464))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp462, tmp463);
  }tmp469 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp464,"(r_init[454] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp469 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[454] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp469);
    }
  }
  (data->simulationInfo->realParameter[958] /* omega_c[454] PARAM */) = sqrt(tmp469);
  TRACE_POP
}

/*
equation index: 13095
type: SIMPLE_ASSIGN
r_init[453] = r_min + 453.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13095};
  (data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (453.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13096
type: SIMPLE_ASSIGN
omega_c[453] = sqrt(G * Md / (r_init[453] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13096(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13096};
  modelica_real tmp470;
  modelica_real tmp471;
  modelica_real tmp472;
  modelica_real tmp473;
  modelica_real tmp474;
  modelica_real tmp475;
  modelica_real tmp476;
  modelica_real tmp477;
  modelica_real tmp478;
  modelica_real tmp479;
  tmp470 = (data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */);
  tmp471 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp472 = (tmp470 * tmp470) + (tmp471 * tmp471);
  tmp473 = 1.5;
  if(tmp472 < 0.0 && tmp473 != 0.0)
  {
    tmp475 = modf(tmp473, &tmp476);
    
    if(tmp475 > 0.5)
    {
      tmp475 -= 1.0;
      tmp476 += 1.0;
    }
    else if(tmp475 < -0.5)
    {
      tmp475 += 1.0;
      tmp476 -= 1.0;
    }
    
    if(fabs(tmp475) < 1e-10)
      tmp474 = pow(tmp472, tmp476);
    else
    {
      tmp478 = modf(1.0/tmp473, &tmp477);
      if(tmp478 > 0.5)
      {
        tmp478 -= 1.0;
        tmp477 += 1.0;
      }
      else if(tmp478 < -0.5)
      {
        tmp478 += 1.0;
        tmp477 -= 1.0;
      }
      if(fabs(tmp478) < 1e-10 && ((unsigned long)tmp477 & 1))
      {
        tmp474 = -pow(-tmp472, tmp475)*pow(tmp472, tmp476);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp472, tmp473);
      }
    }
  }
  else
  {
    tmp474 = pow(tmp472, tmp473);
  }
  if(isnan(tmp474) || isinf(tmp474))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp472, tmp473);
  }tmp479 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp474,"(r_init[453] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp479 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[453] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp479);
    }
  }
  (data->simulationInfo->realParameter[957] /* omega_c[453] PARAM */) = sqrt(tmp479);
  TRACE_POP
}

/*
equation index: 13097
type: SIMPLE_ASSIGN
r_init[452] = r_min + 452.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13097(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13097};
  (data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (452.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13098
type: SIMPLE_ASSIGN
omega_c[452] = sqrt(G * Md / (r_init[452] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13098(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13098};
  modelica_real tmp480;
  modelica_real tmp481;
  modelica_real tmp482;
  modelica_real tmp483;
  modelica_real tmp484;
  modelica_real tmp485;
  modelica_real tmp486;
  modelica_real tmp487;
  modelica_real tmp488;
  modelica_real tmp489;
  tmp480 = (data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */);
  tmp481 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp482 = (tmp480 * tmp480) + (tmp481 * tmp481);
  tmp483 = 1.5;
  if(tmp482 < 0.0 && tmp483 != 0.0)
  {
    tmp485 = modf(tmp483, &tmp486);
    
    if(tmp485 > 0.5)
    {
      tmp485 -= 1.0;
      tmp486 += 1.0;
    }
    else if(tmp485 < -0.5)
    {
      tmp485 += 1.0;
      tmp486 -= 1.0;
    }
    
    if(fabs(tmp485) < 1e-10)
      tmp484 = pow(tmp482, tmp486);
    else
    {
      tmp488 = modf(1.0/tmp483, &tmp487);
      if(tmp488 > 0.5)
      {
        tmp488 -= 1.0;
        tmp487 += 1.0;
      }
      else if(tmp488 < -0.5)
      {
        tmp488 += 1.0;
        tmp487 -= 1.0;
      }
      if(fabs(tmp488) < 1e-10 && ((unsigned long)tmp487 & 1))
      {
        tmp484 = -pow(-tmp482, tmp485)*pow(tmp482, tmp486);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp482, tmp483);
      }
    }
  }
  else
  {
    tmp484 = pow(tmp482, tmp483);
  }
  if(isnan(tmp484) || isinf(tmp484))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp482, tmp483);
  }tmp489 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp484,"(r_init[452] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp489 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[452] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp489);
    }
  }
  (data->simulationInfo->realParameter[956] /* omega_c[452] PARAM */) = sqrt(tmp489);
  TRACE_POP
}

/*
equation index: 13099
type: SIMPLE_ASSIGN
r_init[451] = r_min + 451.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13099};
  (data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (451.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13100
type: SIMPLE_ASSIGN
omega_c[451] = sqrt(G * Md / (r_init[451] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13100};
  modelica_real tmp490;
  modelica_real tmp491;
  modelica_real tmp492;
  modelica_real tmp493;
  modelica_real tmp494;
  modelica_real tmp495;
  modelica_real tmp496;
  modelica_real tmp497;
  modelica_real tmp498;
  modelica_real tmp499;
  tmp490 = (data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */);
  tmp491 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp492 = (tmp490 * tmp490) + (tmp491 * tmp491);
  tmp493 = 1.5;
  if(tmp492 < 0.0 && tmp493 != 0.0)
  {
    tmp495 = modf(tmp493, &tmp496);
    
    if(tmp495 > 0.5)
    {
      tmp495 -= 1.0;
      tmp496 += 1.0;
    }
    else if(tmp495 < -0.5)
    {
      tmp495 += 1.0;
      tmp496 -= 1.0;
    }
    
    if(fabs(tmp495) < 1e-10)
      tmp494 = pow(tmp492, tmp496);
    else
    {
      tmp498 = modf(1.0/tmp493, &tmp497);
      if(tmp498 > 0.5)
      {
        tmp498 -= 1.0;
        tmp497 += 1.0;
      }
      else if(tmp498 < -0.5)
      {
        tmp498 += 1.0;
        tmp497 -= 1.0;
      }
      if(fabs(tmp498) < 1e-10 && ((unsigned long)tmp497 & 1))
      {
        tmp494 = -pow(-tmp492, tmp495)*pow(tmp492, tmp496);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp492, tmp493);
      }
    }
  }
  else
  {
    tmp494 = pow(tmp492, tmp493);
  }
  if(isnan(tmp494) || isinf(tmp494))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp492, tmp493);
  }tmp499 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp494,"(r_init[451] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp499 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[451] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp499);
    }
  }
  (data->simulationInfo->realParameter[955] /* omega_c[451] PARAM */) = sqrt(tmp499);
  TRACE_POP
}

/*
equation index: 13101
type: SIMPLE_ASSIGN
r_init[450] = r_min + 450.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13101};
  (data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (450.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13102
type: SIMPLE_ASSIGN
omega_c[450] = sqrt(G * Md / (r_init[450] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13102(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13102};
  modelica_real tmp500;
  modelica_real tmp501;
  modelica_real tmp502;
  modelica_real tmp503;
  modelica_real tmp504;
  modelica_real tmp505;
  modelica_real tmp506;
  modelica_real tmp507;
  modelica_real tmp508;
  modelica_real tmp509;
  tmp500 = (data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */);
  tmp501 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp502 = (tmp500 * tmp500) + (tmp501 * tmp501);
  tmp503 = 1.5;
  if(tmp502 < 0.0 && tmp503 != 0.0)
  {
    tmp505 = modf(tmp503, &tmp506);
    
    if(tmp505 > 0.5)
    {
      tmp505 -= 1.0;
      tmp506 += 1.0;
    }
    else if(tmp505 < -0.5)
    {
      tmp505 += 1.0;
      tmp506 -= 1.0;
    }
    
    if(fabs(tmp505) < 1e-10)
      tmp504 = pow(tmp502, tmp506);
    else
    {
      tmp508 = modf(1.0/tmp503, &tmp507);
      if(tmp508 > 0.5)
      {
        tmp508 -= 1.0;
        tmp507 += 1.0;
      }
      else if(tmp508 < -0.5)
      {
        tmp508 += 1.0;
        tmp507 -= 1.0;
      }
      if(fabs(tmp508) < 1e-10 && ((unsigned long)tmp507 & 1))
      {
        tmp504 = -pow(-tmp502, tmp505)*pow(tmp502, tmp506);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp502, tmp503);
      }
    }
  }
  else
  {
    tmp504 = pow(tmp502, tmp503);
  }
  if(isnan(tmp504) || isinf(tmp504))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp502, tmp503);
  }tmp509 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp504,"(r_init[450] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp509 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[450] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp509);
    }
  }
  (data->simulationInfo->realParameter[954] /* omega_c[450] PARAM */) = sqrt(tmp509);
  TRACE_POP
}

/*
equation index: 13103
type: SIMPLE_ASSIGN
r_init[449] = r_min + 449.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13103};
  (data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (449.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13104
type: SIMPLE_ASSIGN
omega_c[449] = sqrt(G * Md / (r_init[449] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13104(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13104};
  modelica_real tmp510;
  modelica_real tmp511;
  modelica_real tmp512;
  modelica_real tmp513;
  modelica_real tmp514;
  modelica_real tmp515;
  modelica_real tmp516;
  modelica_real tmp517;
  modelica_real tmp518;
  modelica_real tmp519;
  tmp510 = (data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */);
  tmp511 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp512 = (tmp510 * tmp510) + (tmp511 * tmp511);
  tmp513 = 1.5;
  if(tmp512 < 0.0 && tmp513 != 0.0)
  {
    tmp515 = modf(tmp513, &tmp516);
    
    if(tmp515 > 0.5)
    {
      tmp515 -= 1.0;
      tmp516 += 1.0;
    }
    else if(tmp515 < -0.5)
    {
      tmp515 += 1.0;
      tmp516 -= 1.0;
    }
    
    if(fabs(tmp515) < 1e-10)
      tmp514 = pow(tmp512, tmp516);
    else
    {
      tmp518 = modf(1.0/tmp513, &tmp517);
      if(tmp518 > 0.5)
      {
        tmp518 -= 1.0;
        tmp517 += 1.0;
      }
      else if(tmp518 < -0.5)
      {
        tmp518 += 1.0;
        tmp517 -= 1.0;
      }
      if(fabs(tmp518) < 1e-10 && ((unsigned long)tmp517 & 1))
      {
        tmp514 = -pow(-tmp512, tmp515)*pow(tmp512, tmp516);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp512, tmp513);
      }
    }
  }
  else
  {
    tmp514 = pow(tmp512, tmp513);
  }
  if(isnan(tmp514) || isinf(tmp514))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp512, tmp513);
  }tmp519 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp514,"(r_init[449] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp519 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[449] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp519);
    }
  }
  (data->simulationInfo->realParameter[953] /* omega_c[449] PARAM */) = sqrt(tmp519);
  TRACE_POP
}

/*
equation index: 13105
type: SIMPLE_ASSIGN
r_init[448] = r_min + 448.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13105};
  (data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (448.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13106
type: SIMPLE_ASSIGN
omega_c[448] = sqrt(G * Md / (r_init[448] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13106(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13106};
  modelica_real tmp520;
  modelica_real tmp521;
  modelica_real tmp522;
  modelica_real tmp523;
  modelica_real tmp524;
  modelica_real tmp525;
  modelica_real tmp526;
  modelica_real tmp527;
  modelica_real tmp528;
  modelica_real tmp529;
  tmp520 = (data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */);
  tmp521 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp522 = (tmp520 * tmp520) + (tmp521 * tmp521);
  tmp523 = 1.5;
  if(tmp522 < 0.0 && tmp523 != 0.0)
  {
    tmp525 = modf(tmp523, &tmp526);
    
    if(tmp525 > 0.5)
    {
      tmp525 -= 1.0;
      tmp526 += 1.0;
    }
    else if(tmp525 < -0.5)
    {
      tmp525 += 1.0;
      tmp526 -= 1.0;
    }
    
    if(fabs(tmp525) < 1e-10)
      tmp524 = pow(tmp522, tmp526);
    else
    {
      tmp528 = modf(1.0/tmp523, &tmp527);
      if(tmp528 > 0.5)
      {
        tmp528 -= 1.0;
        tmp527 += 1.0;
      }
      else if(tmp528 < -0.5)
      {
        tmp528 += 1.0;
        tmp527 -= 1.0;
      }
      if(fabs(tmp528) < 1e-10 && ((unsigned long)tmp527 & 1))
      {
        tmp524 = -pow(-tmp522, tmp525)*pow(tmp522, tmp526);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp522, tmp523);
      }
    }
  }
  else
  {
    tmp524 = pow(tmp522, tmp523);
  }
  if(isnan(tmp524) || isinf(tmp524))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp522, tmp523);
  }tmp529 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp524,"(r_init[448] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp529 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[448] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp529);
    }
  }
  (data->simulationInfo->realParameter[952] /* omega_c[448] PARAM */) = sqrt(tmp529);
  TRACE_POP
}

/*
equation index: 13107
type: SIMPLE_ASSIGN
r_init[447] = r_min + 447.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13107};
  (data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (447.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13108
type: SIMPLE_ASSIGN
omega_c[447] = sqrt(G * Md / (r_init[447] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13108};
  modelica_real tmp530;
  modelica_real tmp531;
  modelica_real tmp532;
  modelica_real tmp533;
  modelica_real tmp534;
  modelica_real tmp535;
  modelica_real tmp536;
  modelica_real tmp537;
  modelica_real tmp538;
  modelica_real tmp539;
  tmp530 = (data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */);
  tmp531 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp532 = (tmp530 * tmp530) + (tmp531 * tmp531);
  tmp533 = 1.5;
  if(tmp532 < 0.0 && tmp533 != 0.0)
  {
    tmp535 = modf(tmp533, &tmp536);
    
    if(tmp535 > 0.5)
    {
      tmp535 -= 1.0;
      tmp536 += 1.0;
    }
    else if(tmp535 < -0.5)
    {
      tmp535 += 1.0;
      tmp536 -= 1.0;
    }
    
    if(fabs(tmp535) < 1e-10)
      tmp534 = pow(tmp532, tmp536);
    else
    {
      tmp538 = modf(1.0/tmp533, &tmp537);
      if(tmp538 > 0.5)
      {
        tmp538 -= 1.0;
        tmp537 += 1.0;
      }
      else if(tmp538 < -0.5)
      {
        tmp538 += 1.0;
        tmp537 -= 1.0;
      }
      if(fabs(tmp538) < 1e-10 && ((unsigned long)tmp537 & 1))
      {
        tmp534 = -pow(-tmp532, tmp535)*pow(tmp532, tmp536);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp532, tmp533);
      }
    }
  }
  else
  {
    tmp534 = pow(tmp532, tmp533);
  }
  if(isnan(tmp534) || isinf(tmp534))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp532, tmp533);
  }tmp539 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp534,"(r_init[447] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp539 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[447] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp539);
    }
  }
  (data->simulationInfo->realParameter[951] /* omega_c[447] PARAM */) = sqrt(tmp539);
  TRACE_POP
}

/*
equation index: 13109
type: SIMPLE_ASSIGN
r_init[446] = r_min + 446.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13109};
  (data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (446.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13110
type: SIMPLE_ASSIGN
omega_c[446] = sqrt(G * Md / (r_init[446] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13110(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13110};
  modelica_real tmp540;
  modelica_real tmp541;
  modelica_real tmp542;
  modelica_real tmp543;
  modelica_real tmp544;
  modelica_real tmp545;
  modelica_real tmp546;
  modelica_real tmp547;
  modelica_real tmp548;
  modelica_real tmp549;
  tmp540 = (data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */);
  tmp541 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp542 = (tmp540 * tmp540) + (tmp541 * tmp541);
  tmp543 = 1.5;
  if(tmp542 < 0.0 && tmp543 != 0.0)
  {
    tmp545 = modf(tmp543, &tmp546);
    
    if(tmp545 > 0.5)
    {
      tmp545 -= 1.0;
      tmp546 += 1.0;
    }
    else if(tmp545 < -0.5)
    {
      tmp545 += 1.0;
      tmp546 -= 1.0;
    }
    
    if(fabs(tmp545) < 1e-10)
      tmp544 = pow(tmp542, tmp546);
    else
    {
      tmp548 = modf(1.0/tmp543, &tmp547);
      if(tmp548 > 0.5)
      {
        tmp548 -= 1.0;
        tmp547 += 1.0;
      }
      else if(tmp548 < -0.5)
      {
        tmp548 += 1.0;
        tmp547 -= 1.0;
      }
      if(fabs(tmp548) < 1e-10 && ((unsigned long)tmp547 & 1))
      {
        tmp544 = -pow(-tmp542, tmp545)*pow(tmp542, tmp546);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp542, tmp543);
      }
    }
  }
  else
  {
    tmp544 = pow(tmp542, tmp543);
  }
  if(isnan(tmp544) || isinf(tmp544))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp542, tmp543);
  }tmp549 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp544,"(r_init[446] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp549 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[446] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp549);
    }
  }
  (data->simulationInfo->realParameter[950] /* omega_c[446] PARAM */) = sqrt(tmp549);
  TRACE_POP
}

/*
equation index: 13111
type: SIMPLE_ASSIGN
r_init[445] = r_min + 445.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13111};
  (data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (445.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13112
type: SIMPLE_ASSIGN
omega_c[445] = sqrt(G * Md / (r_init[445] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13112(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13112};
  modelica_real tmp550;
  modelica_real tmp551;
  modelica_real tmp552;
  modelica_real tmp553;
  modelica_real tmp554;
  modelica_real tmp555;
  modelica_real tmp556;
  modelica_real tmp557;
  modelica_real tmp558;
  modelica_real tmp559;
  tmp550 = (data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */);
  tmp551 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp552 = (tmp550 * tmp550) + (tmp551 * tmp551);
  tmp553 = 1.5;
  if(tmp552 < 0.0 && tmp553 != 0.0)
  {
    tmp555 = modf(tmp553, &tmp556);
    
    if(tmp555 > 0.5)
    {
      tmp555 -= 1.0;
      tmp556 += 1.0;
    }
    else if(tmp555 < -0.5)
    {
      tmp555 += 1.0;
      tmp556 -= 1.0;
    }
    
    if(fabs(tmp555) < 1e-10)
      tmp554 = pow(tmp552, tmp556);
    else
    {
      tmp558 = modf(1.0/tmp553, &tmp557);
      if(tmp558 > 0.5)
      {
        tmp558 -= 1.0;
        tmp557 += 1.0;
      }
      else if(tmp558 < -0.5)
      {
        tmp558 += 1.0;
        tmp557 -= 1.0;
      }
      if(fabs(tmp558) < 1e-10 && ((unsigned long)tmp557 & 1))
      {
        tmp554 = -pow(-tmp552, tmp555)*pow(tmp552, tmp556);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp552, tmp553);
      }
    }
  }
  else
  {
    tmp554 = pow(tmp552, tmp553);
  }
  if(isnan(tmp554) || isinf(tmp554))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp552, tmp553);
  }tmp559 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp554,"(r_init[445] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp559 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[445] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp559);
    }
  }
  (data->simulationInfo->realParameter[949] /* omega_c[445] PARAM */) = sqrt(tmp559);
  TRACE_POP
}

/*
equation index: 13113
type: SIMPLE_ASSIGN
r_init[444] = r_min + 444.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13113};
  (data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (444.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13114
type: SIMPLE_ASSIGN
omega_c[444] = sqrt(G * Md / (r_init[444] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13114(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13114};
  modelica_real tmp560;
  modelica_real tmp561;
  modelica_real tmp562;
  modelica_real tmp563;
  modelica_real tmp564;
  modelica_real tmp565;
  modelica_real tmp566;
  modelica_real tmp567;
  modelica_real tmp568;
  modelica_real tmp569;
  tmp560 = (data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */);
  tmp561 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp562 = (tmp560 * tmp560) + (tmp561 * tmp561);
  tmp563 = 1.5;
  if(tmp562 < 0.0 && tmp563 != 0.0)
  {
    tmp565 = modf(tmp563, &tmp566);
    
    if(tmp565 > 0.5)
    {
      tmp565 -= 1.0;
      tmp566 += 1.0;
    }
    else if(tmp565 < -0.5)
    {
      tmp565 += 1.0;
      tmp566 -= 1.0;
    }
    
    if(fabs(tmp565) < 1e-10)
      tmp564 = pow(tmp562, tmp566);
    else
    {
      tmp568 = modf(1.0/tmp563, &tmp567);
      if(tmp568 > 0.5)
      {
        tmp568 -= 1.0;
        tmp567 += 1.0;
      }
      else if(tmp568 < -0.5)
      {
        tmp568 += 1.0;
        tmp567 -= 1.0;
      }
      if(fabs(tmp568) < 1e-10 && ((unsigned long)tmp567 & 1))
      {
        tmp564 = -pow(-tmp562, tmp565)*pow(tmp562, tmp566);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp562, tmp563);
      }
    }
  }
  else
  {
    tmp564 = pow(tmp562, tmp563);
  }
  if(isnan(tmp564) || isinf(tmp564))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp562, tmp563);
  }tmp569 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp564,"(r_init[444] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp569 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[444] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp569);
    }
  }
  (data->simulationInfo->realParameter[948] /* omega_c[444] PARAM */) = sqrt(tmp569);
  TRACE_POP
}

/*
equation index: 13115
type: SIMPLE_ASSIGN
r_init[443] = r_min + 443.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13115};
  (data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (443.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13116
type: SIMPLE_ASSIGN
omega_c[443] = sqrt(G * Md / (r_init[443] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13116};
  modelica_real tmp570;
  modelica_real tmp571;
  modelica_real tmp572;
  modelica_real tmp573;
  modelica_real tmp574;
  modelica_real tmp575;
  modelica_real tmp576;
  modelica_real tmp577;
  modelica_real tmp578;
  modelica_real tmp579;
  tmp570 = (data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */);
  tmp571 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp572 = (tmp570 * tmp570) + (tmp571 * tmp571);
  tmp573 = 1.5;
  if(tmp572 < 0.0 && tmp573 != 0.0)
  {
    tmp575 = modf(tmp573, &tmp576);
    
    if(tmp575 > 0.5)
    {
      tmp575 -= 1.0;
      tmp576 += 1.0;
    }
    else if(tmp575 < -0.5)
    {
      tmp575 += 1.0;
      tmp576 -= 1.0;
    }
    
    if(fabs(tmp575) < 1e-10)
      tmp574 = pow(tmp572, tmp576);
    else
    {
      tmp578 = modf(1.0/tmp573, &tmp577);
      if(tmp578 > 0.5)
      {
        tmp578 -= 1.0;
        tmp577 += 1.0;
      }
      else if(tmp578 < -0.5)
      {
        tmp578 += 1.0;
        tmp577 -= 1.0;
      }
      if(fabs(tmp578) < 1e-10 && ((unsigned long)tmp577 & 1))
      {
        tmp574 = -pow(-tmp572, tmp575)*pow(tmp572, tmp576);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp572, tmp573);
      }
    }
  }
  else
  {
    tmp574 = pow(tmp572, tmp573);
  }
  if(isnan(tmp574) || isinf(tmp574))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp572, tmp573);
  }tmp579 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp574,"(r_init[443] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp579 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[443] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp579);
    }
  }
  (data->simulationInfo->realParameter[947] /* omega_c[443] PARAM */) = sqrt(tmp579);
  TRACE_POP
}

/*
equation index: 13117
type: SIMPLE_ASSIGN
r_init[442] = r_min + 442.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13117};
  (data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (442.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13118
type: SIMPLE_ASSIGN
omega_c[442] = sqrt(G * Md / (r_init[442] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13118(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13118};
  modelica_real tmp580;
  modelica_real tmp581;
  modelica_real tmp582;
  modelica_real tmp583;
  modelica_real tmp584;
  modelica_real tmp585;
  modelica_real tmp586;
  modelica_real tmp587;
  modelica_real tmp588;
  modelica_real tmp589;
  tmp580 = (data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */);
  tmp581 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp582 = (tmp580 * tmp580) + (tmp581 * tmp581);
  tmp583 = 1.5;
  if(tmp582 < 0.0 && tmp583 != 0.0)
  {
    tmp585 = modf(tmp583, &tmp586);
    
    if(tmp585 > 0.5)
    {
      tmp585 -= 1.0;
      tmp586 += 1.0;
    }
    else if(tmp585 < -0.5)
    {
      tmp585 += 1.0;
      tmp586 -= 1.0;
    }
    
    if(fabs(tmp585) < 1e-10)
      tmp584 = pow(tmp582, tmp586);
    else
    {
      tmp588 = modf(1.0/tmp583, &tmp587);
      if(tmp588 > 0.5)
      {
        tmp588 -= 1.0;
        tmp587 += 1.0;
      }
      else if(tmp588 < -0.5)
      {
        tmp588 += 1.0;
        tmp587 -= 1.0;
      }
      if(fabs(tmp588) < 1e-10 && ((unsigned long)tmp587 & 1))
      {
        tmp584 = -pow(-tmp582, tmp585)*pow(tmp582, tmp586);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp582, tmp583);
      }
    }
  }
  else
  {
    tmp584 = pow(tmp582, tmp583);
  }
  if(isnan(tmp584) || isinf(tmp584))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp582, tmp583);
  }tmp589 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp584,"(r_init[442] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp589 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[442] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp589);
    }
  }
  (data->simulationInfo->realParameter[946] /* omega_c[442] PARAM */) = sqrt(tmp589);
  TRACE_POP
}

/*
equation index: 13119
type: SIMPLE_ASSIGN
r_init[441] = r_min + 441.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13119};
  (data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (441.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13120
type: SIMPLE_ASSIGN
omega_c[441] = sqrt(G * Md / (r_init[441] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13120(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13120};
  modelica_real tmp590;
  modelica_real tmp591;
  modelica_real tmp592;
  modelica_real tmp593;
  modelica_real tmp594;
  modelica_real tmp595;
  modelica_real tmp596;
  modelica_real tmp597;
  modelica_real tmp598;
  modelica_real tmp599;
  tmp590 = (data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */);
  tmp591 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp592 = (tmp590 * tmp590) + (tmp591 * tmp591);
  tmp593 = 1.5;
  if(tmp592 < 0.0 && tmp593 != 0.0)
  {
    tmp595 = modf(tmp593, &tmp596);
    
    if(tmp595 > 0.5)
    {
      tmp595 -= 1.0;
      tmp596 += 1.0;
    }
    else if(tmp595 < -0.5)
    {
      tmp595 += 1.0;
      tmp596 -= 1.0;
    }
    
    if(fabs(tmp595) < 1e-10)
      tmp594 = pow(tmp592, tmp596);
    else
    {
      tmp598 = modf(1.0/tmp593, &tmp597);
      if(tmp598 > 0.5)
      {
        tmp598 -= 1.0;
        tmp597 += 1.0;
      }
      else if(tmp598 < -0.5)
      {
        tmp598 += 1.0;
        tmp597 -= 1.0;
      }
      if(fabs(tmp598) < 1e-10 && ((unsigned long)tmp597 & 1))
      {
        tmp594 = -pow(-tmp592, tmp595)*pow(tmp592, tmp596);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp592, tmp593);
      }
    }
  }
  else
  {
    tmp594 = pow(tmp592, tmp593);
  }
  if(isnan(tmp594) || isinf(tmp594))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp592, tmp593);
  }tmp599 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp594,"(r_init[441] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp599 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[441] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp599);
    }
  }
  (data->simulationInfo->realParameter[945] /* omega_c[441] PARAM */) = sqrt(tmp599);
  TRACE_POP
}

/*
equation index: 13121
type: SIMPLE_ASSIGN
r_init[440] = r_min + 440.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13121};
  (data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (440.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13122
type: SIMPLE_ASSIGN
omega_c[440] = sqrt(G * Md / (r_init[440] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13122(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13122};
  modelica_real tmp600;
  modelica_real tmp601;
  modelica_real tmp602;
  modelica_real tmp603;
  modelica_real tmp604;
  modelica_real tmp605;
  modelica_real tmp606;
  modelica_real tmp607;
  modelica_real tmp608;
  modelica_real tmp609;
  tmp600 = (data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */);
  tmp601 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp602 = (tmp600 * tmp600) + (tmp601 * tmp601);
  tmp603 = 1.5;
  if(tmp602 < 0.0 && tmp603 != 0.0)
  {
    tmp605 = modf(tmp603, &tmp606);
    
    if(tmp605 > 0.5)
    {
      tmp605 -= 1.0;
      tmp606 += 1.0;
    }
    else if(tmp605 < -0.5)
    {
      tmp605 += 1.0;
      tmp606 -= 1.0;
    }
    
    if(fabs(tmp605) < 1e-10)
      tmp604 = pow(tmp602, tmp606);
    else
    {
      tmp608 = modf(1.0/tmp603, &tmp607);
      if(tmp608 > 0.5)
      {
        tmp608 -= 1.0;
        tmp607 += 1.0;
      }
      else if(tmp608 < -0.5)
      {
        tmp608 += 1.0;
        tmp607 -= 1.0;
      }
      if(fabs(tmp608) < 1e-10 && ((unsigned long)tmp607 & 1))
      {
        tmp604 = -pow(-tmp602, tmp605)*pow(tmp602, tmp606);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp602, tmp603);
      }
    }
  }
  else
  {
    tmp604 = pow(tmp602, tmp603);
  }
  if(isnan(tmp604) || isinf(tmp604))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp602, tmp603);
  }tmp609 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp604,"(r_init[440] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp609 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[440] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp609);
    }
  }
  (data->simulationInfo->realParameter[944] /* omega_c[440] PARAM */) = sqrt(tmp609);
  TRACE_POP
}

/*
equation index: 13123
type: SIMPLE_ASSIGN
r_init[439] = r_min + 439.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13123};
  (data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (439.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13124
type: SIMPLE_ASSIGN
omega_c[439] = sqrt(G * Md / (r_init[439] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13124};
  modelica_real tmp610;
  modelica_real tmp611;
  modelica_real tmp612;
  modelica_real tmp613;
  modelica_real tmp614;
  modelica_real tmp615;
  modelica_real tmp616;
  modelica_real tmp617;
  modelica_real tmp618;
  modelica_real tmp619;
  tmp610 = (data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */);
  tmp611 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp612 = (tmp610 * tmp610) + (tmp611 * tmp611);
  tmp613 = 1.5;
  if(tmp612 < 0.0 && tmp613 != 0.0)
  {
    tmp615 = modf(tmp613, &tmp616);
    
    if(tmp615 > 0.5)
    {
      tmp615 -= 1.0;
      tmp616 += 1.0;
    }
    else if(tmp615 < -0.5)
    {
      tmp615 += 1.0;
      tmp616 -= 1.0;
    }
    
    if(fabs(tmp615) < 1e-10)
      tmp614 = pow(tmp612, tmp616);
    else
    {
      tmp618 = modf(1.0/tmp613, &tmp617);
      if(tmp618 > 0.5)
      {
        tmp618 -= 1.0;
        tmp617 += 1.0;
      }
      else if(tmp618 < -0.5)
      {
        tmp618 += 1.0;
        tmp617 -= 1.0;
      }
      if(fabs(tmp618) < 1e-10 && ((unsigned long)tmp617 & 1))
      {
        tmp614 = -pow(-tmp612, tmp615)*pow(tmp612, tmp616);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp612, tmp613);
      }
    }
  }
  else
  {
    tmp614 = pow(tmp612, tmp613);
  }
  if(isnan(tmp614) || isinf(tmp614))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp612, tmp613);
  }tmp619 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp614,"(r_init[439] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp619 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[439] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp619);
    }
  }
  (data->simulationInfo->realParameter[943] /* omega_c[439] PARAM */) = sqrt(tmp619);
  TRACE_POP
}

/*
equation index: 13125
type: SIMPLE_ASSIGN
r_init[438] = r_min + 438.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13125};
  (data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (438.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13126
type: SIMPLE_ASSIGN
omega_c[438] = sqrt(G * Md / (r_init[438] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13126(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13126};
  modelica_real tmp620;
  modelica_real tmp621;
  modelica_real tmp622;
  modelica_real tmp623;
  modelica_real tmp624;
  modelica_real tmp625;
  modelica_real tmp626;
  modelica_real tmp627;
  modelica_real tmp628;
  modelica_real tmp629;
  tmp620 = (data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */);
  tmp621 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp622 = (tmp620 * tmp620) + (tmp621 * tmp621);
  tmp623 = 1.5;
  if(tmp622 < 0.0 && tmp623 != 0.0)
  {
    tmp625 = modf(tmp623, &tmp626);
    
    if(tmp625 > 0.5)
    {
      tmp625 -= 1.0;
      tmp626 += 1.0;
    }
    else if(tmp625 < -0.5)
    {
      tmp625 += 1.0;
      tmp626 -= 1.0;
    }
    
    if(fabs(tmp625) < 1e-10)
      tmp624 = pow(tmp622, tmp626);
    else
    {
      tmp628 = modf(1.0/tmp623, &tmp627);
      if(tmp628 > 0.5)
      {
        tmp628 -= 1.0;
        tmp627 += 1.0;
      }
      else if(tmp628 < -0.5)
      {
        tmp628 += 1.0;
        tmp627 -= 1.0;
      }
      if(fabs(tmp628) < 1e-10 && ((unsigned long)tmp627 & 1))
      {
        tmp624 = -pow(-tmp622, tmp625)*pow(tmp622, tmp626);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp622, tmp623);
      }
    }
  }
  else
  {
    tmp624 = pow(tmp622, tmp623);
  }
  if(isnan(tmp624) || isinf(tmp624))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp622, tmp623);
  }tmp629 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp624,"(r_init[438] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp629 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[438] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp629);
    }
  }
  (data->simulationInfo->realParameter[942] /* omega_c[438] PARAM */) = sqrt(tmp629);
  TRACE_POP
}

/*
equation index: 13127
type: SIMPLE_ASSIGN
r_init[437] = r_min + 437.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13127};
  (data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (437.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13128
type: SIMPLE_ASSIGN
omega_c[437] = sqrt(G * Md / (r_init[437] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13128(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13128};
  modelica_real tmp630;
  modelica_real tmp631;
  modelica_real tmp632;
  modelica_real tmp633;
  modelica_real tmp634;
  modelica_real tmp635;
  modelica_real tmp636;
  modelica_real tmp637;
  modelica_real tmp638;
  modelica_real tmp639;
  tmp630 = (data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */);
  tmp631 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp632 = (tmp630 * tmp630) + (tmp631 * tmp631);
  tmp633 = 1.5;
  if(tmp632 < 0.0 && tmp633 != 0.0)
  {
    tmp635 = modf(tmp633, &tmp636);
    
    if(tmp635 > 0.5)
    {
      tmp635 -= 1.0;
      tmp636 += 1.0;
    }
    else if(tmp635 < -0.5)
    {
      tmp635 += 1.0;
      tmp636 -= 1.0;
    }
    
    if(fabs(tmp635) < 1e-10)
      tmp634 = pow(tmp632, tmp636);
    else
    {
      tmp638 = modf(1.0/tmp633, &tmp637);
      if(tmp638 > 0.5)
      {
        tmp638 -= 1.0;
        tmp637 += 1.0;
      }
      else if(tmp638 < -0.5)
      {
        tmp638 += 1.0;
        tmp637 -= 1.0;
      }
      if(fabs(tmp638) < 1e-10 && ((unsigned long)tmp637 & 1))
      {
        tmp634 = -pow(-tmp632, tmp635)*pow(tmp632, tmp636);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp632, tmp633);
      }
    }
  }
  else
  {
    tmp634 = pow(tmp632, tmp633);
  }
  if(isnan(tmp634) || isinf(tmp634))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp632, tmp633);
  }tmp639 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp634,"(r_init[437] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp639 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[437] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp639);
    }
  }
  (data->simulationInfo->realParameter[941] /* omega_c[437] PARAM */) = sqrt(tmp639);
  TRACE_POP
}

/*
equation index: 13129
type: SIMPLE_ASSIGN
r_init[436] = r_min + 436.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13129};
  (data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (436.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13130
type: SIMPLE_ASSIGN
omega_c[436] = sqrt(G * Md / (r_init[436] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13130(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13130};
  modelica_real tmp640;
  modelica_real tmp641;
  modelica_real tmp642;
  modelica_real tmp643;
  modelica_real tmp644;
  modelica_real tmp645;
  modelica_real tmp646;
  modelica_real tmp647;
  modelica_real tmp648;
  modelica_real tmp649;
  tmp640 = (data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */);
  tmp641 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp642 = (tmp640 * tmp640) + (tmp641 * tmp641);
  tmp643 = 1.5;
  if(tmp642 < 0.0 && tmp643 != 0.0)
  {
    tmp645 = modf(tmp643, &tmp646);
    
    if(tmp645 > 0.5)
    {
      tmp645 -= 1.0;
      tmp646 += 1.0;
    }
    else if(tmp645 < -0.5)
    {
      tmp645 += 1.0;
      tmp646 -= 1.0;
    }
    
    if(fabs(tmp645) < 1e-10)
      tmp644 = pow(tmp642, tmp646);
    else
    {
      tmp648 = modf(1.0/tmp643, &tmp647);
      if(tmp648 > 0.5)
      {
        tmp648 -= 1.0;
        tmp647 += 1.0;
      }
      else if(tmp648 < -0.5)
      {
        tmp648 += 1.0;
        tmp647 -= 1.0;
      }
      if(fabs(tmp648) < 1e-10 && ((unsigned long)tmp647 & 1))
      {
        tmp644 = -pow(-tmp642, tmp645)*pow(tmp642, tmp646);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp642, tmp643);
      }
    }
  }
  else
  {
    tmp644 = pow(tmp642, tmp643);
  }
  if(isnan(tmp644) || isinf(tmp644))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp642, tmp643);
  }tmp649 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp644,"(r_init[436] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp649 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[436] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp649);
    }
  }
  (data->simulationInfo->realParameter[940] /* omega_c[436] PARAM */) = sqrt(tmp649);
  TRACE_POP
}

/*
equation index: 13131
type: SIMPLE_ASSIGN
r_init[435] = r_min + 435.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13131};
  (data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (435.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13132
type: SIMPLE_ASSIGN
omega_c[435] = sqrt(G * Md / (r_init[435] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13132};
  modelica_real tmp650;
  modelica_real tmp651;
  modelica_real tmp652;
  modelica_real tmp653;
  modelica_real tmp654;
  modelica_real tmp655;
  modelica_real tmp656;
  modelica_real tmp657;
  modelica_real tmp658;
  modelica_real tmp659;
  tmp650 = (data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */);
  tmp651 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp652 = (tmp650 * tmp650) + (tmp651 * tmp651);
  tmp653 = 1.5;
  if(tmp652 < 0.0 && tmp653 != 0.0)
  {
    tmp655 = modf(tmp653, &tmp656);
    
    if(tmp655 > 0.5)
    {
      tmp655 -= 1.0;
      tmp656 += 1.0;
    }
    else if(tmp655 < -0.5)
    {
      tmp655 += 1.0;
      tmp656 -= 1.0;
    }
    
    if(fabs(tmp655) < 1e-10)
      tmp654 = pow(tmp652, tmp656);
    else
    {
      tmp658 = modf(1.0/tmp653, &tmp657);
      if(tmp658 > 0.5)
      {
        tmp658 -= 1.0;
        tmp657 += 1.0;
      }
      else if(tmp658 < -0.5)
      {
        tmp658 += 1.0;
        tmp657 -= 1.0;
      }
      if(fabs(tmp658) < 1e-10 && ((unsigned long)tmp657 & 1))
      {
        tmp654 = -pow(-tmp652, tmp655)*pow(tmp652, tmp656);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp652, tmp653);
      }
    }
  }
  else
  {
    tmp654 = pow(tmp652, tmp653);
  }
  if(isnan(tmp654) || isinf(tmp654))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp652, tmp653);
  }tmp659 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp654,"(r_init[435] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp659 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[435] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp659);
    }
  }
  (data->simulationInfo->realParameter[939] /* omega_c[435] PARAM */) = sqrt(tmp659);
  TRACE_POP
}

/*
equation index: 13133
type: SIMPLE_ASSIGN
r_init[434] = r_min + 434.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13133};
  (data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (434.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13134
type: SIMPLE_ASSIGN
omega_c[434] = sqrt(G * Md / (r_init[434] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13134(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13134};
  modelica_real tmp660;
  modelica_real tmp661;
  modelica_real tmp662;
  modelica_real tmp663;
  modelica_real tmp664;
  modelica_real tmp665;
  modelica_real tmp666;
  modelica_real tmp667;
  modelica_real tmp668;
  modelica_real tmp669;
  tmp660 = (data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */);
  tmp661 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp662 = (tmp660 * tmp660) + (tmp661 * tmp661);
  tmp663 = 1.5;
  if(tmp662 < 0.0 && tmp663 != 0.0)
  {
    tmp665 = modf(tmp663, &tmp666);
    
    if(tmp665 > 0.5)
    {
      tmp665 -= 1.0;
      tmp666 += 1.0;
    }
    else if(tmp665 < -0.5)
    {
      tmp665 += 1.0;
      tmp666 -= 1.0;
    }
    
    if(fabs(tmp665) < 1e-10)
      tmp664 = pow(tmp662, tmp666);
    else
    {
      tmp668 = modf(1.0/tmp663, &tmp667);
      if(tmp668 > 0.5)
      {
        tmp668 -= 1.0;
        tmp667 += 1.0;
      }
      else if(tmp668 < -0.5)
      {
        tmp668 += 1.0;
        tmp667 -= 1.0;
      }
      if(fabs(tmp668) < 1e-10 && ((unsigned long)tmp667 & 1))
      {
        tmp664 = -pow(-tmp662, tmp665)*pow(tmp662, tmp666);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp662, tmp663);
      }
    }
  }
  else
  {
    tmp664 = pow(tmp662, tmp663);
  }
  if(isnan(tmp664) || isinf(tmp664))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp662, tmp663);
  }tmp669 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp664,"(r_init[434] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp669 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[434] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp669);
    }
  }
  (data->simulationInfo->realParameter[938] /* omega_c[434] PARAM */) = sqrt(tmp669);
  TRACE_POP
}

/*
equation index: 13135
type: SIMPLE_ASSIGN
r_init[433] = r_min + 433.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13135};
  (data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (433.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13136
type: SIMPLE_ASSIGN
omega_c[433] = sqrt(G * Md / (r_init[433] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13136(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13136};
  modelica_real tmp670;
  modelica_real tmp671;
  modelica_real tmp672;
  modelica_real tmp673;
  modelica_real tmp674;
  modelica_real tmp675;
  modelica_real tmp676;
  modelica_real tmp677;
  modelica_real tmp678;
  modelica_real tmp679;
  tmp670 = (data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */);
  tmp671 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp672 = (tmp670 * tmp670) + (tmp671 * tmp671);
  tmp673 = 1.5;
  if(tmp672 < 0.0 && tmp673 != 0.0)
  {
    tmp675 = modf(tmp673, &tmp676);
    
    if(tmp675 > 0.5)
    {
      tmp675 -= 1.0;
      tmp676 += 1.0;
    }
    else if(tmp675 < -0.5)
    {
      tmp675 += 1.0;
      tmp676 -= 1.0;
    }
    
    if(fabs(tmp675) < 1e-10)
      tmp674 = pow(tmp672, tmp676);
    else
    {
      tmp678 = modf(1.0/tmp673, &tmp677);
      if(tmp678 > 0.5)
      {
        tmp678 -= 1.0;
        tmp677 += 1.0;
      }
      else if(tmp678 < -0.5)
      {
        tmp678 += 1.0;
        tmp677 -= 1.0;
      }
      if(fabs(tmp678) < 1e-10 && ((unsigned long)tmp677 & 1))
      {
        tmp674 = -pow(-tmp672, tmp675)*pow(tmp672, tmp676);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp672, tmp673);
      }
    }
  }
  else
  {
    tmp674 = pow(tmp672, tmp673);
  }
  if(isnan(tmp674) || isinf(tmp674))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp672, tmp673);
  }tmp679 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp674,"(r_init[433] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp679 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[433] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp679);
    }
  }
  (data->simulationInfo->realParameter[937] /* omega_c[433] PARAM */) = sqrt(tmp679);
  TRACE_POP
}

/*
equation index: 13137
type: SIMPLE_ASSIGN
r_init[432] = r_min + 432.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13137};
  (data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (432.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13138
type: SIMPLE_ASSIGN
omega_c[432] = sqrt(G * Md / (r_init[432] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13138(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13138};
  modelica_real tmp680;
  modelica_real tmp681;
  modelica_real tmp682;
  modelica_real tmp683;
  modelica_real tmp684;
  modelica_real tmp685;
  modelica_real tmp686;
  modelica_real tmp687;
  modelica_real tmp688;
  modelica_real tmp689;
  tmp680 = (data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */);
  tmp681 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp682 = (tmp680 * tmp680) + (tmp681 * tmp681);
  tmp683 = 1.5;
  if(tmp682 < 0.0 && tmp683 != 0.0)
  {
    tmp685 = modf(tmp683, &tmp686);
    
    if(tmp685 > 0.5)
    {
      tmp685 -= 1.0;
      tmp686 += 1.0;
    }
    else if(tmp685 < -0.5)
    {
      tmp685 += 1.0;
      tmp686 -= 1.0;
    }
    
    if(fabs(tmp685) < 1e-10)
      tmp684 = pow(tmp682, tmp686);
    else
    {
      tmp688 = modf(1.0/tmp683, &tmp687);
      if(tmp688 > 0.5)
      {
        tmp688 -= 1.0;
        tmp687 += 1.0;
      }
      else if(tmp688 < -0.5)
      {
        tmp688 += 1.0;
        tmp687 -= 1.0;
      }
      if(fabs(tmp688) < 1e-10 && ((unsigned long)tmp687 & 1))
      {
        tmp684 = -pow(-tmp682, tmp685)*pow(tmp682, tmp686);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp682, tmp683);
      }
    }
  }
  else
  {
    tmp684 = pow(tmp682, tmp683);
  }
  if(isnan(tmp684) || isinf(tmp684))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp682, tmp683);
  }tmp689 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp684,"(r_init[432] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp689 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[432] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp689);
    }
  }
  (data->simulationInfo->realParameter[936] /* omega_c[432] PARAM */) = sqrt(tmp689);
  TRACE_POP
}

/*
equation index: 13139
type: SIMPLE_ASSIGN
r_init[431] = r_min + 431.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13139(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13139};
  (data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (431.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13140
type: SIMPLE_ASSIGN
omega_c[431] = sqrt(G * Md / (r_init[431] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13140};
  modelica_real tmp690;
  modelica_real tmp691;
  modelica_real tmp692;
  modelica_real tmp693;
  modelica_real tmp694;
  modelica_real tmp695;
  modelica_real tmp696;
  modelica_real tmp697;
  modelica_real tmp698;
  modelica_real tmp699;
  tmp690 = (data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */);
  tmp691 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp692 = (tmp690 * tmp690) + (tmp691 * tmp691);
  tmp693 = 1.5;
  if(tmp692 < 0.0 && tmp693 != 0.0)
  {
    tmp695 = modf(tmp693, &tmp696);
    
    if(tmp695 > 0.5)
    {
      tmp695 -= 1.0;
      tmp696 += 1.0;
    }
    else if(tmp695 < -0.5)
    {
      tmp695 += 1.0;
      tmp696 -= 1.0;
    }
    
    if(fabs(tmp695) < 1e-10)
      tmp694 = pow(tmp692, tmp696);
    else
    {
      tmp698 = modf(1.0/tmp693, &tmp697);
      if(tmp698 > 0.5)
      {
        tmp698 -= 1.0;
        tmp697 += 1.0;
      }
      else if(tmp698 < -0.5)
      {
        tmp698 += 1.0;
        tmp697 -= 1.0;
      }
      if(fabs(tmp698) < 1e-10 && ((unsigned long)tmp697 & 1))
      {
        tmp694 = -pow(-tmp692, tmp695)*pow(tmp692, tmp696);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp692, tmp693);
      }
    }
  }
  else
  {
    tmp694 = pow(tmp692, tmp693);
  }
  if(isnan(tmp694) || isinf(tmp694))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp692, tmp693);
  }tmp699 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp694,"(r_init[431] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp699 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[431] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp699);
    }
  }
  (data->simulationInfo->realParameter[935] /* omega_c[431] PARAM */) = sqrt(tmp699);
  TRACE_POP
}

/*
equation index: 13141
type: SIMPLE_ASSIGN
r_init[430] = r_min + 430.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13141};
  (data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (430.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13142
type: SIMPLE_ASSIGN
omega_c[430] = sqrt(G * Md / (r_init[430] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13142(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13142};
  modelica_real tmp700;
  modelica_real tmp701;
  modelica_real tmp702;
  modelica_real tmp703;
  modelica_real tmp704;
  modelica_real tmp705;
  modelica_real tmp706;
  modelica_real tmp707;
  modelica_real tmp708;
  modelica_real tmp709;
  tmp700 = (data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */);
  tmp701 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp702 = (tmp700 * tmp700) + (tmp701 * tmp701);
  tmp703 = 1.5;
  if(tmp702 < 0.0 && tmp703 != 0.0)
  {
    tmp705 = modf(tmp703, &tmp706);
    
    if(tmp705 > 0.5)
    {
      tmp705 -= 1.0;
      tmp706 += 1.0;
    }
    else if(tmp705 < -0.5)
    {
      tmp705 += 1.0;
      tmp706 -= 1.0;
    }
    
    if(fabs(tmp705) < 1e-10)
      tmp704 = pow(tmp702, tmp706);
    else
    {
      tmp708 = modf(1.0/tmp703, &tmp707);
      if(tmp708 > 0.5)
      {
        tmp708 -= 1.0;
        tmp707 += 1.0;
      }
      else if(tmp708 < -0.5)
      {
        tmp708 += 1.0;
        tmp707 -= 1.0;
      }
      if(fabs(tmp708) < 1e-10 && ((unsigned long)tmp707 & 1))
      {
        tmp704 = -pow(-tmp702, tmp705)*pow(tmp702, tmp706);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp702, tmp703);
      }
    }
  }
  else
  {
    tmp704 = pow(tmp702, tmp703);
  }
  if(isnan(tmp704) || isinf(tmp704))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp702, tmp703);
  }tmp709 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp704,"(r_init[430] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp709 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[430] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp709);
    }
  }
  (data->simulationInfo->realParameter[934] /* omega_c[430] PARAM */) = sqrt(tmp709);
  TRACE_POP
}

/*
equation index: 13143
type: SIMPLE_ASSIGN
r_init[429] = r_min + 429.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13143};
  (data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (429.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13144
type: SIMPLE_ASSIGN
omega_c[429] = sqrt(G * Md / (r_init[429] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13144(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13144};
  modelica_real tmp710;
  modelica_real tmp711;
  modelica_real tmp712;
  modelica_real tmp713;
  modelica_real tmp714;
  modelica_real tmp715;
  modelica_real tmp716;
  modelica_real tmp717;
  modelica_real tmp718;
  modelica_real tmp719;
  tmp710 = (data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */);
  tmp711 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp712 = (tmp710 * tmp710) + (tmp711 * tmp711);
  tmp713 = 1.5;
  if(tmp712 < 0.0 && tmp713 != 0.0)
  {
    tmp715 = modf(tmp713, &tmp716);
    
    if(tmp715 > 0.5)
    {
      tmp715 -= 1.0;
      tmp716 += 1.0;
    }
    else if(tmp715 < -0.5)
    {
      tmp715 += 1.0;
      tmp716 -= 1.0;
    }
    
    if(fabs(tmp715) < 1e-10)
      tmp714 = pow(tmp712, tmp716);
    else
    {
      tmp718 = modf(1.0/tmp713, &tmp717);
      if(tmp718 > 0.5)
      {
        tmp718 -= 1.0;
        tmp717 += 1.0;
      }
      else if(tmp718 < -0.5)
      {
        tmp718 += 1.0;
        tmp717 -= 1.0;
      }
      if(fabs(tmp718) < 1e-10 && ((unsigned long)tmp717 & 1))
      {
        tmp714 = -pow(-tmp712, tmp715)*pow(tmp712, tmp716);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp712, tmp713);
      }
    }
  }
  else
  {
    tmp714 = pow(tmp712, tmp713);
  }
  if(isnan(tmp714) || isinf(tmp714))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp712, tmp713);
  }tmp719 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp714,"(r_init[429] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp719 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[429] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp719);
    }
  }
  (data->simulationInfo->realParameter[933] /* omega_c[429] PARAM */) = sqrt(tmp719);
  TRACE_POP
}

/*
equation index: 13145
type: SIMPLE_ASSIGN
r_init[428] = r_min + 428.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13145};
  (data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (428.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13146
type: SIMPLE_ASSIGN
omega_c[428] = sqrt(G * Md / (r_init[428] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13146(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13146};
  modelica_real tmp720;
  modelica_real tmp721;
  modelica_real tmp722;
  modelica_real tmp723;
  modelica_real tmp724;
  modelica_real tmp725;
  modelica_real tmp726;
  modelica_real tmp727;
  modelica_real tmp728;
  modelica_real tmp729;
  tmp720 = (data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */);
  tmp721 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp722 = (tmp720 * tmp720) + (tmp721 * tmp721);
  tmp723 = 1.5;
  if(tmp722 < 0.0 && tmp723 != 0.0)
  {
    tmp725 = modf(tmp723, &tmp726);
    
    if(tmp725 > 0.5)
    {
      tmp725 -= 1.0;
      tmp726 += 1.0;
    }
    else if(tmp725 < -0.5)
    {
      tmp725 += 1.0;
      tmp726 -= 1.0;
    }
    
    if(fabs(tmp725) < 1e-10)
      tmp724 = pow(tmp722, tmp726);
    else
    {
      tmp728 = modf(1.0/tmp723, &tmp727);
      if(tmp728 > 0.5)
      {
        tmp728 -= 1.0;
        tmp727 += 1.0;
      }
      else if(tmp728 < -0.5)
      {
        tmp728 += 1.0;
        tmp727 -= 1.0;
      }
      if(fabs(tmp728) < 1e-10 && ((unsigned long)tmp727 & 1))
      {
        tmp724 = -pow(-tmp722, tmp725)*pow(tmp722, tmp726);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp722, tmp723);
      }
    }
  }
  else
  {
    tmp724 = pow(tmp722, tmp723);
  }
  if(isnan(tmp724) || isinf(tmp724))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp722, tmp723);
  }tmp729 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp724,"(r_init[428] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp729 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[428] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp729);
    }
  }
  (data->simulationInfo->realParameter[932] /* omega_c[428] PARAM */) = sqrt(tmp729);
  TRACE_POP
}

/*
equation index: 13147
type: SIMPLE_ASSIGN
r_init[427] = r_min + 427.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13147};
  (data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (427.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13148
type: SIMPLE_ASSIGN
omega_c[427] = sqrt(G * Md / (r_init[427] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13148};
  modelica_real tmp730;
  modelica_real tmp731;
  modelica_real tmp732;
  modelica_real tmp733;
  modelica_real tmp734;
  modelica_real tmp735;
  modelica_real tmp736;
  modelica_real tmp737;
  modelica_real tmp738;
  modelica_real tmp739;
  tmp730 = (data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */);
  tmp731 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp732 = (tmp730 * tmp730) + (tmp731 * tmp731);
  tmp733 = 1.5;
  if(tmp732 < 0.0 && tmp733 != 0.0)
  {
    tmp735 = modf(tmp733, &tmp736);
    
    if(tmp735 > 0.5)
    {
      tmp735 -= 1.0;
      tmp736 += 1.0;
    }
    else if(tmp735 < -0.5)
    {
      tmp735 += 1.0;
      tmp736 -= 1.0;
    }
    
    if(fabs(tmp735) < 1e-10)
      tmp734 = pow(tmp732, tmp736);
    else
    {
      tmp738 = modf(1.0/tmp733, &tmp737);
      if(tmp738 > 0.5)
      {
        tmp738 -= 1.0;
        tmp737 += 1.0;
      }
      else if(tmp738 < -0.5)
      {
        tmp738 += 1.0;
        tmp737 -= 1.0;
      }
      if(fabs(tmp738) < 1e-10 && ((unsigned long)tmp737 & 1))
      {
        tmp734 = -pow(-tmp732, tmp735)*pow(tmp732, tmp736);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp732, tmp733);
      }
    }
  }
  else
  {
    tmp734 = pow(tmp732, tmp733);
  }
  if(isnan(tmp734) || isinf(tmp734))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp732, tmp733);
  }tmp739 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp734,"(r_init[427] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp739 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[427] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp739);
    }
  }
  (data->simulationInfo->realParameter[931] /* omega_c[427] PARAM */) = sqrt(tmp739);
  TRACE_POP
}

/*
equation index: 13149
type: SIMPLE_ASSIGN
r_init[426] = r_min + 426.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13149};
  (data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (426.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13150
type: SIMPLE_ASSIGN
omega_c[426] = sqrt(G * Md / (r_init[426] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13150(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13150};
  modelica_real tmp740;
  modelica_real tmp741;
  modelica_real tmp742;
  modelica_real tmp743;
  modelica_real tmp744;
  modelica_real tmp745;
  modelica_real tmp746;
  modelica_real tmp747;
  modelica_real tmp748;
  modelica_real tmp749;
  tmp740 = (data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */);
  tmp741 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp742 = (tmp740 * tmp740) + (tmp741 * tmp741);
  tmp743 = 1.5;
  if(tmp742 < 0.0 && tmp743 != 0.0)
  {
    tmp745 = modf(tmp743, &tmp746);
    
    if(tmp745 > 0.5)
    {
      tmp745 -= 1.0;
      tmp746 += 1.0;
    }
    else if(tmp745 < -0.5)
    {
      tmp745 += 1.0;
      tmp746 -= 1.0;
    }
    
    if(fabs(tmp745) < 1e-10)
      tmp744 = pow(tmp742, tmp746);
    else
    {
      tmp748 = modf(1.0/tmp743, &tmp747);
      if(tmp748 > 0.5)
      {
        tmp748 -= 1.0;
        tmp747 += 1.0;
      }
      else if(tmp748 < -0.5)
      {
        tmp748 += 1.0;
        tmp747 -= 1.0;
      }
      if(fabs(tmp748) < 1e-10 && ((unsigned long)tmp747 & 1))
      {
        tmp744 = -pow(-tmp742, tmp745)*pow(tmp742, tmp746);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp742, tmp743);
      }
    }
  }
  else
  {
    tmp744 = pow(tmp742, tmp743);
  }
  if(isnan(tmp744) || isinf(tmp744))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp742, tmp743);
  }tmp749 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp744,"(r_init[426] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp749 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[426] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp749);
    }
  }
  (data->simulationInfo->realParameter[930] /* omega_c[426] PARAM */) = sqrt(tmp749);
  TRACE_POP
}

/*
equation index: 13151
type: SIMPLE_ASSIGN
r_init[425] = r_min + 425.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13151};
  (data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (425.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13152
type: SIMPLE_ASSIGN
omega_c[425] = sqrt(G * Md / (r_init[425] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13152(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13152};
  modelica_real tmp750;
  modelica_real tmp751;
  modelica_real tmp752;
  modelica_real tmp753;
  modelica_real tmp754;
  modelica_real tmp755;
  modelica_real tmp756;
  modelica_real tmp757;
  modelica_real tmp758;
  modelica_real tmp759;
  tmp750 = (data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */);
  tmp751 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp752 = (tmp750 * tmp750) + (tmp751 * tmp751);
  tmp753 = 1.5;
  if(tmp752 < 0.0 && tmp753 != 0.0)
  {
    tmp755 = modf(tmp753, &tmp756);
    
    if(tmp755 > 0.5)
    {
      tmp755 -= 1.0;
      tmp756 += 1.0;
    }
    else if(tmp755 < -0.5)
    {
      tmp755 += 1.0;
      tmp756 -= 1.0;
    }
    
    if(fabs(tmp755) < 1e-10)
      tmp754 = pow(tmp752, tmp756);
    else
    {
      tmp758 = modf(1.0/tmp753, &tmp757);
      if(tmp758 > 0.5)
      {
        tmp758 -= 1.0;
        tmp757 += 1.0;
      }
      else if(tmp758 < -0.5)
      {
        tmp758 += 1.0;
        tmp757 -= 1.0;
      }
      if(fabs(tmp758) < 1e-10 && ((unsigned long)tmp757 & 1))
      {
        tmp754 = -pow(-tmp752, tmp755)*pow(tmp752, tmp756);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp752, tmp753);
      }
    }
  }
  else
  {
    tmp754 = pow(tmp752, tmp753);
  }
  if(isnan(tmp754) || isinf(tmp754))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp752, tmp753);
  }tmp759 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp754,"(r_init[425] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp759 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[425] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp759);
    }
  }
  (data->simulationInfo->realParameter[929] /* omega_c[425] PARAM */) = sqrt(tmp759);
  TRACE_POP
}

/*
equation index: 13153
type: SIMPLE_ASSIGN
r_init[424] = r_min + 424.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13153};
  (data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (424.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13154
type: SIMPLE_ASSIGN
omega_c[424] = sqrt(G * Md / (r_init[424] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13154(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13154};
  modelica_real tmp760;
  modelica_real tmp761;
  modelica_real tmp762;
  modelica_real tmp763;
  modelica_real tmp764;
  modelica_real tmp765;
  modelica_real tmp766;
  modelica_real tmp767;
  modelica_real tmp768;
  modelica_real tmp769;
  tmp760 = (data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */);
  tmp761 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp762 = (tmp760 * tmp760) + (tmp761 * tmp761);
  tmp763 = 1.5;
  if(tmp762 < 0.0 && tmp763 != 0.0)
  {
    tmp765 = modf(tmp763, &tmp766);
    
    if(tmp765 > 0.5)
    {
      tmp765 -= 1.0;
      tmp766 += 1.0;
    }
    else if(tmp765 < -0.5)
    {
      tmp765 += 1.0;
      tmp766 -= 1.0;
    }
    
    if(fabs(tmp765) < 1e-10)
      tmp764 = pow(tmp762, tmp766);
    else
    {
      tmp768 = modf(1.0/tmp763, &tmp767);
      if(tmp768 > 0.5)
      {
        tmp768 -= 1.0;
        tmp767 += 1.0;
      }
      else if(tmp768 < -0.5)
      {
        tmp768 += 1.0;
        tmp767 -= 1.0;
      }
      if(fabs(tmp768) < 1e-10 && ((unsigned long)tmp767 & 1))
      {
        tmp764 = -pow(-tmp762, tmp765)*pow(tmp762, tmp766);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp762, tmp763);
      }
    }
  }
  else
  {
    tmp764 = pow(tmp762, tmp763);
  }
  if(isnan(tmp764) || isinf(tmp764))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp762, tmp763);
  }tmp769 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp764,"(r_init[424] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp769 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[424] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp769);
    }
  }
  (data->simulationInfo->realParameter[928] /* omega_c[424] PARAM */) = sqrt(tmp769);
  TRACE_POP
}

/*
equation index: 13155
type: SIMPLE_ASSIGN
r_init[423] = r_min + 423.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13155};
  (data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (423.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13156
type: SIMPLE_ASSIGN
omega_c[423] = sqrt(G * Md / (r_init[423] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13156};
  modelica_real tmp770;
  modelica_real tmp771;
  modelica_real tmp772;
  modelica_real tmp773;
  modelica_real tmp774;
  modelica_real tmp775;
  modelica_real tmp776;
  modelica_real tmp777;
  modelica_real tmp778;
  modelica_real tmp779;
  tmp770 = (data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */);
  tmp771 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp772 = (tmp770 * tmp770) + (tmp771 * tmp771);
  tmp773 = 1.5;
  if(tmp772 < 0.0 && tmp773 != 0.0)
  {
    tmp775 = modf(tmp773, &tmp776);
    
    if(tmp775 > 0.5)
    {
      tmp775 -= 1.0;
      tmp776 += 1.0;
    }
    else if(tmp775 < -0.5)
    {
      tmp775 += 1.0;
      tmp776 -= 1.0;
    }
    
    if(fabs(tmp775) < 1e-10)
      tmp774 = pow(tmp772, tmp776);
    else
    {
      tmp778 = modf(1.0/tmp773, &tmp777);
      if(tmp778 > 0.5)
      {
        tmp778 -= 1.0;
        tmp777 += 1.0;
      }
      else if(tmp778 < -0.5)
      {
        tmp778 += 1.0;
        tmp777 -= 1.0;
      }
      if(fabs(tmp778) < 1e-10 && ((unsigned long)tmp777 & 1))
      {
        tmp774 = -pow(-tmp772, tmp775)*pow(tmp772, tmp776);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp772, tmp773);
      }
    }
  }
  else
  {
    tmp774 = pow(tmp772, tmp773);
  }
  if(isnan(tmp774) || isinf(tmp774))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp772, tmp773);
  }tmp779 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp774,"(r_init[423] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp779 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[423] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp779);
    }
  }
  (data->simulationInfo->realParameter[927] /* omega_c[423] PARAM */) = sqrt(tmp779);
  TRACE_POP
}

/*
equation index: 13157
type: SIMPLE_ASSIGN
r_init[422] = r_min + 422.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13157};
  (data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (422.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13158
type: SIMPLE_ASSIGN
omega_c[422] = sqrt(G * Md / (r_init[422] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13158(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13158};
  modelica_real tmp780;
  modelica_real tmp781;
  modelica_real tmp782;
  modelica_real tmp783;
  modelica_real tmp784;
  modelica_real tmp785;
  modelica_real tmp786;
  modelica_real tmp787;
  modelica_real tmp788;
  modelica_real tmp789;
  tmp780 = (data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */);
  tmp781 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp782 = (tmp780 * tmp780) + (tmp781 * tmp781);
  tmp783 = 1.5;
  if(tmp782 < 0.0 && tmp783 != 0.0)
  {
    tmp785 = modf(tmp783, &tmp786);
    
    if(tmp785 > 0.5)
    {
      tmp785 -= 1.0;
      tmp786 += 1.0;
    }
    else if(tmp785 < -0.5)
    {
      tmp785 += 1.0;
      tmp786 -= 1.0;
    }
    
    if(fabs(tmp785) < 1e-10)
      tmp784 = pow(tmp782, tmp786);
    else
    {
      tmp788 = modf(1.0/tmp783, &tmp787);
      if(tmp788 > 0.5)
      {
        tmp788 -= 1.0;
        tmp787 += 1.0;
      }
      else if(tmp788 < -0.5)
      {
        tmp788 += 1.0;
        tmp787 -= 1.0;
      }
      if(fabs(tmp788) < 1e-10 && ((unsigned long)tmp787 & 1))
      {
        tmp784 = -pow(-tmp782, tmp785)*pow(tmp782, tmp786);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp782, tmp783);
      }
    }
  }
  else
  {
    tmp784 = pow(tmp782, tmp783);
  }
  if(isnan(tmp784) || isinf(tmp784))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp782, tmp783);
  }tmp789 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp784,"(r_init[422] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp789 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[422] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp789);
    }
  }
  (data->simulationInfo->realParameter[926] /* omega_c[422] PARAM */) = sqrt(tmp789);
  TRACE_POP
}

/*
equation index: 13159
type: SIMPLE_ASSIGN
r_init[421] = r_min + 421.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13159};
  (data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (421.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13160
type: SIMPLE_ASSIGN
omega_c[421] = sqrt(G * Md / (r_init[421] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13160(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13160};
  modelica_real tmp790;
  modelica_real tmp791;
  modelica_real tmp792;
  modelica_real tmp793;
  modelica_real tmp794;
  modelica_real tmp795;
  modelica_real tmp796;
  modelica_real tmp797;
  modelica_real tmp798;
  modelica_real tmp799;
  tmp790 = (data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */);
  tmp791 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp792 = (tmp790 * tmp790) + (tmp791 * tmp791);
  tmp793 = 1.5;
  if(tmp792 < 0.0 && tmp793 != 0.0)
  {
    tmp795 = modf(tmp793, &tmp796);
    
    if(tmp795 > 0.5)
    {
      tmp795 -= 1.0;
      tmp796 += 1.0;
    }
    else if(tmp795 < -0.5)
    {
      tmp795 += 1.0;
      tmp796 -= 1.0;
    }
    
    if(fabs(tmp795) < 1e-10)
      tmp794 = pow(tmp792, tmp796);
    else
    {
      tmp798 = modf(1.0/tmp793, &tmp797);
      if(tmp798 > 0.5)
      {
        tmp798 -= 1.0;
        tmp797 += 1.0;
      }
      else if(tmp798 < -0.5)
      {
        tmp798 += 1.0;
        tmp797 -= 1.0;
      }
      if(fabs(tmp798) < 1e-10 && ((unsigned long)tmp797 & 1))
      {
        tmp794 = -pow(-tmp792, tmp795)*pow(tmp792, tmp796);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp792, tmp793);
      }
    }
  }
  else
  {
    tmp794 = pow(tmp792, tmp793);
  }
  if(isnan(tmp794) || isinf(tmp794))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp792, tmp793);
  }tmp799 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp794,"(r_init[421] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp799 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[421] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp799);
    }
  }
  (data->simulationInfo->realParameter[925] /* omega_c[421] PARAM */) = sqrt(tmp799);
  TRACE_POP
}

/*
equation index: 13161
type: SIMPLE_ASSIGN
r_init[420] = r_min + 420.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13161};
  (data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (420.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13162
type: SIMPLE_ASSIGN
omega_c[420] = sqrt(G * Md / (r_init[420] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13162(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13162};
  modelica_real tmp800;
  modelica_real tmp801;
  modelica_real tmp802;
  modelica_real tmp803;
  modelica_real tmp804;
  modelica_real tmp805;
  modelica_real tmp806;
  modelica_real tmp807;
  modelica_real tmp808;
  modelica_real tmp809;
  tmp800 = (data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */);
  tmp801 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp802 = (tmp800 * tmp800) + (tmp801 * tmp801);
  tmp803 = 1.5;
  if(tmp802 < 0.0 && tmp803 != 0.0)
  {
    tmp805 = modf(tmp803, &tmp806);
    
    if(tmp805 > 0.5)
    {
      tmp805 -= 1.0;
      tmp806 += 1.0;
    }
    else if(tmp805 < -0.5)
    {
      tmp805 += 1.0;
      tmp806 -= 1.0;
    }
    
    if(fabs(tmp805) < 1e-10)
      tmp804 = pow(tmp802, tmp806);
    else
    {
      tmp808 = modf(1.0/tmp803, &tmp807);
      if(tmp808 > 0.5)
      {
        tmp808 -= 1.0;
        tmp807 += 1.0;
      }
      else if(tmp808 < -0.5)
      {
        tmp808 += 1.0;
        tmp807 -= 1.0;
      }
      if(fabs(tmp808) < 1e-10 && ((unsigned long)tmp807 & 1))
      {
        tmp804 = -pow(-tmp802, tmp805)*pow(tmp802, tmp806);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp802, tmp803);
      }
    }
  }
  else
  {
    tmp804 = pow(tmp802, tmp803);
  }
  if(isnan(tmp804) || isinf(tmp804))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp802, tmp803);
  }tmp809 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp804,"(r_init[420] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp809 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[420] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp809);
    }
  }
  (data->simulationInfo->realParameter[924] /* omega_c[420] PARAM */) = sqrt(tmp809);
  TRACE_POP
}

/*
equation index: 13163
type: SIMPLE_ASSIGN
r_init[419] = r_min + 419.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13163};
  (data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (419.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13164
type: SIMPLE_ASSIGN
omega_c[419] = sqrt(G * Md / (r_init[419] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13164};
  modelica_real tmp810;
  modelica_real tmp811;
  modelica_real tmp812;
  modelica_real tmp813;
  modelica_real tmp814;
  modelica_real tmp815;
  modelica_real tmp816;
  modelica_real tmp817;
  modelica_real tmp818;
  modelica_real tmp819;
  tmp810 = (data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */);
  tmp811 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp812 = (tmp810 * tmp810) + (tmp811 * tmp811);
  tmp813 = 1.5;
  if(tmp812 < 0.0 && tmp813 != 0.0)
  {
    tmp815 = modf(tmp813, &tmp816);
    
    if(tmp815 > 0.5)
    {
      tmp815 -= 1.0;
      tmp816 += 1.0;
    }
    else if(tmp815 < -0.5)
    {
      tmp815 += 1.0;
      tmp816 -= 1.0;
    }
    
    if(fabs(tmp815) < 1e-10)
      tmp814 = pow(tmp812, tmp816);
    else
    {
      tmp818 = modf(1.0/tmp813, &tmp817);
      if(tmp818 > 0.5)
      {
        tmp818 -= 1.0;
        tmp817 += 1.0;
      }
      else if(tmp818 < -0.5)
      {
        tmp818 += 1.0;
        tmp817 -= 1.0;
      }
      if(fabs(tmp818) < 1e-10 && ((unsigned long)tmp817 & 1))
      {
        tmp814 = -pow(-tmp812, tmp815)*pow(tmp812, tmp816);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp812, tmp813);
      }
    }
  }
  else
  {
    tmp814 = pow(tmp812, tmp813);
  }
  if(isnan(tmp814) || isinf(tmp814))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp812, tmp813);
  }tmp819 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp814,"(r_init[419] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp819 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[419] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp819);
    }
  }
  (data->simulationInfo->realParameter[923] /* omega_c[419] PARAM */) = sqrt(tmp819);
  TRACE_POP
}

/*
equation index: 13165
type: SIMPLE_ASSIGN
r_init[418] = r_min + 418.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13165};
  (data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (418.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13166
type: SIMPLE_ASSIGN
omega_c[418] = sqrt(G * Md / (r_init[418] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13166(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13166};
  modelica_real tmp820;
  modelica_real tmp821;
  modelica_real tmp822;
  modelica_real tmp823;
  modelica_real tmp824;
  modelica_real tmp825;
  modelica_real tmp826;
  modelica_real tmp827;
  modelica_real tmp828;
  modelica_real tmp829;
  tmp820 = (data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */);
  tmp821 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp822 = (tmp820 * tmp820) + (tmp821 * tmp821);
  tmp823 = 1.5;
  if(tmp822 < 0.0 && tmp823 != 0.0)
  {
    tmp825 = modf(tmp823, &tmp826);
    
    if(tmp825 > 0.5)
    {
      tmp825 -= 1.0;
      tmp826 += 1.0;
    }
    else if(tmp825 < -0.5)
    {
      tmp825 += 1.0;
      tmp826 -= 1.0;
    }
    
    if(fabs(tmp825) < 1e-10)
      tmp824 = pow(tmp822, tmp826);
    else
    {
      tmp828 = modf(1.0/tmp823, &tmp827);
      if(tmp828 > 0.5)
      {
        tmp828 -= 1.0;
        tmp827 += 1.0;
      }
      else if(tmp828 < -0.5)
      {
        tmp828 += 1.0;
        tmp827 -= 1.0;
      }
      if(fabs(tmp828) < 1e-10 && ((unsigned long)tmp827 & 1))
      {
        tmp824 = -pow(-tmp822, tmp825)*pow(tmp822, tmp826);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp822, tmp823);
      }
    }
  }
  else
  {
    tmp824 = pow(tmp822, tmp823);
  }
  if(isnan(tmp824) || isinf(tmp824))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp822, tmp823);
  }tmp829 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp824,"(r_init[418] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp829 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[418] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp829);
    }
  }
  (data->simulationInfo->realParameter[922] /* omega_c[418] PARAM */) = sqrt(tmp829);
  TRACE_POP
}

/*
equation index: 13167
type: SIMPLE_ASSIGN
r_init[417] = r_min + 417.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13167};
  (data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (417.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13168
type: SIMPLE_ASSIGN
omega_c[417] = sqrt(G * Md / (r_init[417] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13168(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13168};
  modelica_real tmp830;
  modelica_real tmp831;
  modelica_real tmp832;
  modelica_real tmp833;
  modelica_real tmp834;
  modelica_real tmp835;
  modelica_real tmp836;
  modelica_real tmp837;
  modelica_real tmp838;
  modelica_real tmp839;
  tmp830 = (data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */);
  tmp831 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp832 = (tmp830 * tmp830) + (tmp831 * tmp831);
  tmp833 = 1.5;
  if(tmp832 < 0.0 && tmp833 != 0.0)
  {
    tmp835 = modf(tmp833, &tmp836);
    
    if(tmp835 > 0.5)
    {
      tmp835 -= 1.0;
      tmp836 += 1.0;
    }
    else if(tmp835 < -0.5)
    {
      tmp835 += 1.0;
      tmp836 -= 1.0;
    }
    
    if(fabs(tmp835) < 1e-10)
      tmp834 = pow(tmp832, tmp836);
    else
    {
      tmp838 = modf(1.0/tmp833, &tmp837);
      if(tmp838 > 0.5)
      {
        tmp838 -= 1.0;
        tmp837 += 1.0;
      }
      else if(tmp838 < -0.5)
      {
        tmp838 += 1.0;
        tmp837 -= 1.0;
      }
      if(fabs(tmp838) < 1e-10 && ((unsigned long)tmp837 & 1))
      {
        tmp834 = -pow(-tmp832, tmp835)*pow(tmp832, tmp836);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp832, tmp833);
      }
    }
  }
  else
  {
    tmp834 = pow(tmp832, tmp833);
  }
  if(isnan(tmp834) || isinf(tmp834))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp832, tmp833);
  }tmp839 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp834,"(r_init[417] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp839 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[417] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp839);
    }
  }
  (data->simulationInfo->realParameter[921] /* omega_c[417] PARAM */) = sqrt(tmp839);
  TRACE_POP
}

/*
equation index: 13169
type: SIMPLE_ASSIGN
r_init[416] = r_min + 416.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13169};
  (data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (416.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13170
type: SIMPLE_ASSIGN
omega_c[416] = sqrt(G * Md / (r_init[416] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13170(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13170};
  modelica_real tmp840;
  modelica_real tmp841;
  modelica_real tmp842;
  modelica_real tmp843;
  modelica_real tmp844;
  modelica_real tmp845;
  modelica_real tmp846;
  modelica_real tmp847;
  modelica_real tmp848;
  modelica_real tmp849;
  tmp840 = (data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */);
  tmp841 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp842 = (tmp840 * tmp840) + (tmp841 * tmp841);
  tmp843 = 1.5;
  if(tmp842 < 0.0 && tmp843 != 0.0)
  {
    tmp845 = modf(tmp843, &tmp846);
    
    if(tmp845 > 0.5)
    {
      tmp845 -= 1.0;
      tmp846 += 1.0;
    }
    else if(tmp845 < -0.5)
    {
      tmp845 += 1.0;
      tmp846 -= 1.0;
    }
    
    if(fabs(tmp845) < 1e-10)
      tmp844 = pow(tmp842, tmp846);
    else
    {
      tmp848 = modf(1.0/tmp843, &tmp847);
      if(tmp848 > 0.5)
      {
        tmp848 -= 1.0;
        tmp847 += 1.0;
      }
      else if(tmp848 < -0.5)
      {
        tmp848 += 1.0;
        tmp847 -= 1.0;
      }
      if(fabs(tmp848) < 1e-10 && ((unsigned long)tmp847 & 1))
      {
        tmp844 = -pow(-tmp842, tmp845)*pow(tmp842, tmp846);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp842, tmp843);
      }
    }
  }
  else
  {
    tmp844 = pow(tmp842, tmp843);
  }
  if(isnan(tmp844) || isinf(tmp844))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp842, tmp843);
  }tmp849 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp844,"(r_init[416] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp849 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[416] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp849);
    }
  }
  (data->simulationInfo->realParameter[920] /* omega_c[416] PARAM */) = sqrt(tmp849);
  TRACE_POP
}

/*
equation index: 13171
type: SIMPLE_ASSIGN
r_init[415] = r_min + 415.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13171};
  (data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (415.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13172
type: SIMPLE_ASSIGN
omega_c[415] = sqrt(G * Md / (r_init[415] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13172};
  modelica_real tmp850;
  modelica_real tmp851;
  modelica_real tmp852;
  modelica_real tmp853;
  modelica_real tmp854;
  modelica_real tmp855;
  modelica_real tmp856;
  modelica_real tmp857;
  modelica_real tmp858;
  modelica_real tmp859;
  tmp850 = (data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */);
  tmp851 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp852 = (tmp850 * tmp850) + (tmp851 * tmp851);
  tmp853 = 1.5;
  if(tmp852 < 0.0 && tmp853 != 0.0)
  {
    tmp855 = modf(tmp853, &tmp856);
    
    if(tmp855 > 0.5)
    {
      tmp855 -= 1.0;
      tmp856 += 1.0;
    }
    else if(tmp855 < -0.5)
    {
      tmp855 += 1.0;
      tmp856 -= 1.0;
    }
    
    if(fabs(tmp855) < 1e-10)
      tmp854 = pow(tmp852, tmp856);
    else
    {
      tmp858 = modf(1.0/tmp853, &tmp857);
      if(tmp858 > 0.5)
      {
        tmp858 -= 1.0;
        tmp857 += 1.0;
      }
      else if(tmp858 < -0.5)
      {
        tmp858 += 1.0;
        tmp857 -= 1.0;
      }
      if(fabs(tmp858) < 1e-10 && ((unsigned long)tmp857 & 1))
      {
        tmp854 = -pow(-tmp852, tmp855)*pow(tmp852, tmp856);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp852, tmp853);
      }
    }
  }
  else
  {
    tmp854 = pow(tmp852, tmp853);
  }
  if(isnan(tmp854) || isinf(tmp854))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp852, tmp853);
  }tmp859 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp854,"(r_init[415] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp859 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[415] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp859);
    }
  }
  (data->simulationInfo->realParameter[919] /* omega_c[415] PARAM */) = sqrt(tmp859);
  TRACE_POP
}

/*
equation index: 13173
type: SIMPLE_ASSIGN
r_init[414] = r_min + 414.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13173};
  (data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (414.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13174
type: SIMPLE_ASSIGN
omega_c[414] = sqrt(G * Md / (r_init[414] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13174(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13174};
  modelica_real tmp860;
  modelica_real tmp861;
  modelica_real tmp862;
  modelica_real tmp863;
  modelica_real tmp864;
  modelica_real tmp865;
  modelica_real tmp866;
  modelica_real tmp867;
  modelica_real tmp868;
  modelica_real tmp869;
  tmp860 = (data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */);
  tmp861 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp862 = (tmp860 * tmp860) + (tmp861 * tmp861);
  tmp863 = 1.5;
  if(tmp862 < 0.0 && tmp863 != 0.0)
  {
    tmp865 = modf(tmp863, &tmp866);
    
    if(tmp865 > 0.5)
    {
      tmp865 -= 1.0;
      tmp866 += 1.0;
    }
    else if(tmp865 < -0.5)
    {
      tmp865 += 1.0;
      tmp866 -= 1.0;
    }
    
    if(fabs(tmp865) < 1e-10)
      tmp864 = pow(tmp862, tmp866);
    else
    {
      tmp868 = modf(1.0/tmp863, &tmp867);
      if(tmp868 > 0.5)
      {
        tmp868 -= 1.0;
        tmp867 += 1.0;
      }
      else if(tmp868 < -0.5)
      {
        tmp868 += 1.0;
        tmp867 -= 1.0;
      }
      if(fabs(tmp868) < 1e-10 && ((unsigned long)tmp867 & 1))
      {
        tmp864 = -pow(-tmp862, tmp865)*pow(tmp862, tmp866);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp862, tmp863);
      }
    }
  }
  else
  {
    tmp864 = pow(tmp862, tmp863);
  }
  if(isnan(tmp864) || isinf(tmp864))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp862, tmp863);
  }tmp869 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp864,"(r_init[414] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp869 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[414] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp869);
    }
  }
  (data->simulationInfo->realParameter[918] /* omega_c[414] PARAM */) = sqrt(tmp869);
  TRACE_POP
}

/*
equation index: 13175
type: SIMPLE_ASSIGN
r_init[413] = r_min + 413.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13175};
  (data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (413.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13176
type: SIMPLE_ASSIGN
omega_c[413] = sqrt(G * Md / (r_init[413] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13176(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13176};
  modelica_real tmp870;
  modelica_real tmp871;
  modelica_real tmp872;
  modelica_real tmp873;
  modelica_real tmp874;
  modelica_real tmp875;
  modelica_real tmp876;
  modelica_real tmp877;
  modelica_real tmp878;
  modelica_real tmp879;
  tmp870 = (data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */);
  tmp871 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp872 = (tmp870 * tmp870) + (tmp871 * tmp871);
  tmp873 = 1.5;
  if(tmp872 < 0.0 && tmp873 != 0.0)
  {
    tmp875 = modf(tmp873, &tmp876);
    
    if(tmp875 > 0.5)
    {
      tmp875 -= 1.0;
      tmp876 += 1.0;
    }
    else if(tmp875 < -0.5)
    {
      tmp875 += 1.0;
      tmp876 -= 1.0;
    }
    
    if(fabs(tmp875) < 1e-10)
      tmp874 = pow(tmp872, tmp876);
    else
    {
      tmp878 = modf(1.0/tmp873, &tmp877);
      if(tmp878 > 0.5)
      {
        tmp878 -= 1.0;
        tmp877 += 1.0;
      }
      else if(tmp878 < -0.5)
      {
        tmp878 += 1.0;
        tmp877 -= 1.0;
      }
      if(fabs(tmp878) < 1e-10 && ((unsigned long)tmp877 & 1))
      {
        tmp874 = -pow(-tmp872, tmp875)*pow(tmp872, tmp876);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp872, tmp873);
      }
    }
  }
  else
  {
    tmp874 = pow(tmp872, tmp873);
  }
  if(isnan(tmp874) || isinf(tmp874))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp872, tmp873);
  }tmp879 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp874,"(r_init[413] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp879 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[413] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp879);
    }
  }
  (data->simulationInfo->realParameter[917] /* omega_c[413] PARAM */) = sqrt(tmp879);
  TRACE_POP
}

/*
equation index: 13177
type: SIMPLE_ASSIGN
r_init[412] = r_min + 412.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13177};
  (data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (412.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13178
type: SIMPLE_ASSIGN
omega_c[412] = sqrt(G * Md / (r_init[412] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13178(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13178};
  modelica_real tmp880;
  modelica_real tmp881;
  modelica_real tmp882;
  modelica_real tmp883;
  modelica_real tmp884;
  modelica_real tmp885;
  modelica_real tmp886;
  modelica_real tmp887;
  modelica_real tmp888;
  modelica_real tmp889;
  tmp880 = (data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */);
  tmp881 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp882 = (tmp880 * tmp880) + (tmp881 * tmp881);
  tmp883 = 1.5;
  if(tmp882 < 0.0 && tmp883 != 0.0)
  {
    tmp885 = modf(tmp883, &tmp886);
    
    if(tmp885 > 0.5)
    {
      tmp885 -= 1.0;
      tmp886 += 1.0;
    }
    else if(tmp885 < -0.5)
    {
      tmp885 += 1.0;
      tmp886 -= 1.0;
    }
    
    if(fabs(tmp885) < 1e-10)
      tmp884 = pow(tmp882, tmp886);
    else
    {
      tmp888 = modf(1.0/tmp883, &tmp887);
      if(tmp888 > 0.5)
      {
        tmp888 -= 1.0;
        tmp887 += 1.0;
      }
      else if(tmp888 < -0.5)
      {
        tmp888 += 1.0;
        tmp887 -= 1.0;
      }
      if(fabs(tmp888) < 1e-10 && ((unsigned long)tmp887 & 1))
      {
        tmp884 = -pow(-tmp882, tmp885)*pow(tmp882, tmp886);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp882, tmp883);
      }
    }
  }
  else
  {
    tmp884 = pow(tmp882, tmp883);
  }
  if(isnan(tmp884) || isinf(tmp884))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp882, tmp883);
  }tmp889 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp884,"(r_init[412] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp889 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[412] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp889);
    }
  }
  (data->simulationInfo->realParameter[916] /* omega_c[412] PARAM */) = sqrt(tmp889);
  TRACE_POP
}

/*
equation index: 13179
type: SIMPLE_ASSIGN
r_init[411] = r_min + 411.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13179};
  (data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (411.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13180
type: SIMPLE_ASSIGN
omega_c[411] = sqrt(G * Md / (r_init[411] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13180};
  modelica_real tmp890;
  modelica_real tmp891;
  modelica_real tmp892;
  modelica_real tmp893;
  modelica_real tmp894;
  modelica_real tmp895;
  modelica_real tmp896;
  modelica_real tmp897;
  modelica_real tmp898;
  modelica_real tmp899;
  tmp890 = (data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */);
  tmp891 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp892 = (tmp890 * tmp890) + (tmp891 * tmp891);
  tmp893 = 1.5;
  if(tmp892 < 0.0 && tmp893 != 0.0)
  {
    tmp895 = modf(tmp893, &tmp896);
    
    if(tmp895 > 0.5)
    {
      tmp895 -= 1.0;
      tmp896 += 1.0;
    }
    else if(tmp895 < -0.5)
    {
      tmp895 += 1.0;
      tmp896 -= 1.0;
    }
    
    if(fabs(tmp895) < 1e-10)
      tmp894 = pow(tmp892, tmp896);
    else
    {
      tmp898 = modf(1.0/tmp893, &tmp897);
      if(tmp898 > 0.5)
      {
        tmp898 -= 1.0;
        tmp897 += 1.0;
      }
      else if(tmp898 < -0.5)
      {
        tmp898 += 1.0;
        tmp897 -= 1.0;
      }
      if(fabs(tmp898) < 1e-10 && ((unsigned long)tmp897 & 1))
      {
        tmp894 = -pow(-tmp892, tmp895)*pow(tmp892, tmp896);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp892, tmp893);
      }
    }
  }
  else
  {
    tmp894 = pow(tmp892, tmp893);
  }
  if(isnan(tmp894) || isinf(tmp894))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp892, tmp893);
  }tmp899 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp894,"(r_init[411] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp899 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[411] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp899);
    }
  }
  (data->simulationInfo->realParameter[915] /* omega_c[411] PARAM */) = sqrt(tmp899);
  TRACE_POP
}

/*
equation index: 13181
type: SIMPLE_ASSIGN
r_init[410] = r_min + 410.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13181};
  (data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (410.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13182
type: SIMPLE_ASSIGN
omega_c[410] = sqrt(G * Md / (r_init[410] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13182(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13182};
  modelica_real tmp900;
  modelica_real tmp901;
  modelica_real tmp902;
  modelica_real tmp903;
  modelica_real tmp904;
  modelica_real tmp905;
  modelica_real tmp906;
  modelica_real tmp907;
  modelica_real tmp908;
  modelica_real tmp909;
  tmp900 = (data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */);
  tmp901 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp902 = (tmp900 * tmp900) + (tmp901 * tmp901);
  tmp903 = 1.5;
  if(tmp902 < 0.0 && tmp903 != 0.0)
  {
    tmp905 = modf(tmp903, &tmp906);
    
    if(tmp905 > 0.5)
    {
      tmp905 -= 1.0;
      tmp906 += 1.0;
    }
    else if(tmp905 < -0.5)
    {
      tmp905 += 1.0;
      tmp906 -= 1.0;
    }
    
    if(fabs(tmp905) < 1e-10)
      tmp904 = pow(tmp902, tmp906);
    else
    {
      tmp908 = modf(1.0/tmp903, &tmp907);
      if(tmp908 > 0.5)
      {
        tmp908 -= 1.0;
        tmp907 += 1.0;
      }
      else if(tmp908 < -0.5)
      {
        tmp908 += 1.0;
        tmp907 -= 1.0;
      }
      if(fabs(tmp908) < 1e-10 && ((unsigned long)tmp907 & 1))
      {
        tmp904 = -pow(-tmp902, tmp905)*pow(tmp902, tmp906);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp902, tmp903);
      }
    }
  }
  else
  {
    tmp904 = pow(tmp902, tmp903);
  }
  if(isnan(tmp904) || isinf(tmp904))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp902, tmp903);
  }tmp909 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp904,"(r_init[410] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp909 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[410] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp909);
    }
  }
  (data->simulationInfo->realParameter[914] /* omega_c[410] PARAM */) = sqrt(tmp909);
  TRACE_POP
}

/*
equation index: 13183
type: SIMPLE_ASSIGN
r_init[409] = r_min + 409.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13183};
  (data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (409.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13184
type: SIMPLE_ASSIGN
omega_c[409] = sqrt(G * Md / (r_init[409] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13184(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13184};
  modelica_real tmp910;
  modelica_real tmp911;
  modelica_real tmp912;
  modelica_real tmp913;
  modelica_real tmp914;
  modelica_real tmp915;
  modelica_real tmp916;
  modelica_real tmp917;
  modelica_real tmp918;
  modelica_real tmp919;
  tmp910 = (data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */);
  tmp911 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp912 = (tmp910 * tmp910) + (tmp911 * tmp911);
  tmp913 = 1.5;
  if(tmp912 < 0.0 && tmp913 != 0.0)
  {
    tmp915 = modf(tmp913, &tmp916);
    
    if(tmp915 > 0.5)
    {
      tmp915 -= 1.0;
      tmp916 += 1.0;
    }
    else if(tmp915 < -0.5)
    {
      tmp915 += 1.0;
      tmp916 -= 1.0;
    }
    
    if(fabs(tmp915) < 1e-10)
      tmp914 = pow(tmp912, tmp916);
    else
    {
      tmp918 = modf(1.0/tmp913, &tmp917);
      if(tmp918 > 0.5)
      {
        tmp918 -= 1.0;
        tmp917 += 1.0;
      }
      else if(tmp918 < -0.5)
      {
        tmp918 += 1.0;
        tmp917 -= 1.0;
      }
      if(fabs(tmp918) < 1e-10 && ((unsigned long)tmp917 & 1))
      {
        tmp914 = -pow(-tmp912, tmp915)*pow(tmp912, tmp916);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp912, tmp913);
      }
    }
  }
  else
  {
    tmp914 = pow(tmp912, tmp913);
  }
  if(isnan(tmp914) || isinf(tmp914))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp912, tmp913);
  }tmp919 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp914,"(r_init[409] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp919 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[409] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp919);
    }
  }
  (data->simulationInfo->realParameter[913] /* omega_c[409] PARAM */) = sqrt(tmp919);
  TRACE_POP
}

/*
equation index: 13185
type: SIMPLE_ASSIGN
r_init[408] = r_min + 408.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13185};
  (data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (408.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13186
type: SIMPLE_ASSIGN
omega_c[408] = sqrt(G * Md / (r_init[408] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13186(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13186};
  modelica_real tmp920;
  modelica_real tmp921;
  modelica_real tmp922;
  modelica_real tmp923;
  modelica_real tmp924;
  modelica_real tmp925;
  modelica_real tmp926;
  modelica_real tmp927;
  modelica_real tmp928;
  modelica_real tmp929;
  tmp920 = (data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */);
  tmp921 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp922 = (tmp920 * tmp920) + (tmp921 * tmp921);
  tmp923 = 1.5;
  if(tmp922 < 0.0 && tmp923 != 0.0)
  {
    tmp925 = modf(tmp923, &tmp926);
    
    if(tmp925 > 0.5)
    {
      tmp925 -= 1.0;
      tmp926 += 1.0;
    }
    else if(tmp925 < -0.5)
    {
      tmp925 += 1.0;
      tmp926 -= 1.0;
    }
    
    if(fabs(tmp925) < 1e-10)
      tmp924 = pow(tmp922, tmp926);
    else
    {
      tmp928 = modf(1.0/tmp923, &tmp927);
      if(tmp928 > 0.5)
      {
        tmp928 -= 1.0;
        tmp927 += 1.0;
      }
      else if(tmp928 < -0.5)
      {
        tmp928 += 1.0;
        tmp927 -= 1.0;
      }
      if(fabs(tmp928) < 1e-10 && ((unsigned long)tmp927 & 1))
      {
        tmp924 = -pow(-tmp922, tmp925)*pow(tmp922, tmp926);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp922, tmp923);
      }
    }
  }
  else
  {
    tmp924 = pow(tmp922, tmp923);
  }
  if(isnan(tmp924) || isinf(tmp924))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp922, tmp923);
  }tmp929 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp924,"(r_init[408] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp929 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[408] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp929);
    }
  }
  (data->simulationInfo->realParameter[912] /* omega_c[408] PARAM */) = sqrt(tmp929);
  TRACE_POP
}

/*
equation index: 13187
type: SIMPLE_ASSIGN
r_init[407] = r_min + 407.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13187};
  (data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (407.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13188
type: SIMPLE_ASSIGN
omega_c[407] = sqrt(G * Md / (r_init[407] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13188};
  modelica_real tmp930;
  modelica_real tmp931;
  modelica_real tmp932;
  modelica_real tmp933;
  modelica_real tmp934;
  modelica_real tmp935;
  modelica_real tmp936;
  modelica_real tmp937;
  modelica_real tmp938;
  modelica_real tmp939;
  tmp930 = (data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */);
  tmp931 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp932 = (tmp930 * tmp930) + (tmp931 * tmp931);
  tmp933 = 1.5;
  if(tmp932 < 0.0 && tmp933 != 0.0)
  {
    tmp935 = modf(tmp933, &tmp936);
    
    if(tmp935 > 0.5)
    {
      tmp935 -= 1.0;
      tmp936 += 1.0;
    }
    else if(tmp935 < -0.5)
    {
      tmp935 += 1.0;
      tmp936 -= 1.0;
    }
    
    if(fabs(tmp935) < 1e-10)
      tmp934 = pow(tmp932, tmp936);
    else
    {
      tmp938 = modf(1.0/tmp933, &tmp937);
      if(tmp938 > 0.5)
      {
        tmp938 -= 1.0;
        tmp937 += 1.0;
      }
      else if(tmp938 < -0.5)
      {
        tmp938 += 1.0;
        tmp937 -= 1.0;
      }
      if(fabs(tmp938) < 1e-10 && ((unsigned long)tmp937 & 1))
      {
        tmp934 = -pow(-tmp932, tmp935)*pow(tmp932, tmp936);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp932, tmp933);
      }
    }
  }
  else
  {
    tmp934 = pow(tmp932, tmp933);
  }
  if(isnan(tmp934) || isinf(tmp934))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp932, tmp933);
  }tmp939 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp934,"(r_init[407] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp939 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[407] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp939);
    }
  }
  (data->simulationInfo->realParameter[911] /* omega_c[407] PARAM */) = sqrt(tmp939);
  TRACE_POP
}

/*
equation index: 13189
type: SIMPLE_ASSIGN
r_init[406] = r_min + 406.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13189};
  (data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (406.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13190
type: SIMPLE_ASSIGN
omega_c[406] = sqrt(G * Md / (r_init[406] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13190(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13190};
  modelica_real tmp940;
  modelica_real tmp941;
  modelica_real tmp942;
  modelica_real tmp943;
  modelica_real tmp944;
  modelica_real tmp945;
  modelica_real tmp946;
  modelica_real tmp947;
  modelica_real tmp948;
  modelica_real tmp949;
  tmp940 = (data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */);
  tmp941 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp942 = (tmp940 * tmp940) + (tmp941 * tmp941);
  tmp943 = 1.5;
  if(tmp942 < 0.0 && tmp943 != 0.0)
  {
    tmp945 = modf(tmp943, &tmp946);
    
    if(tmp945 > 0.5)
    {
      tmp945 -= 1.0;
      tmp946 += 1.0;
    }
    else if(tmp945 < -0.5)
    {
      tmp945 += 1.0;
      tmp946 -= 1.0;
    }
    
    if(fabs(tmp945) < 1e-10)
      tmp944 = pow(tmp942, tmp946);
    else
    {
      tmp948 = modf(1.0/tmp943, &tmp947);
      if(tmp948 > 0.5)
      {
        tmp948 -= 1.0;
        tmp947 += 1.0;
      }
      else if(tmp948 < -0.5)
      {
        tmp948 += 1.0;
        tmp947 -= 1.0;
      }
      if(fabs(tmp948) < 1e-10 && ((unsigned long)tmp947 & 1))
      {
        tmp944 = -pow(-tmp942, tmp945)*pow(tmp942, tmp946);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp942, tmp943);
      }
    }
  }
  else
  {
    tmp944 = pow(tmp942, tmp943);
  }
  if(isnan(tmp944) || isinf(tmp944))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp942, tmp943);
  }tmp949 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp944,"(r_init[406] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp949 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[406] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp949);
    }
  }
  (data->simulationInfo->realParameter[910] /* omega_c[406] PARAM */) = sqrt(tmp949);
  TRACE_POP
}

/*
equation index: 13191
type: SIMPLE_ASSIGN
r_init[405] = r_min + 405.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13191};
  (data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (405.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13192
type: SIMPLE_ASSIGN
omega_c[405] = sqrt(G * Md / (r_init[405] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13192(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13192};
  modelica_real tmp950;
  modelica_real tmp951;
  modelica_real tmp952;
  modelica_real tmp953;
  modelica_real tmp954;
  modelica_real tmp955;
  modelica_real tmp956;
  modelica_real tmp957;
  modelica_real tmp958;
  modelica_real tmp959;
  tmp950 = (data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */);
  tmp951 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp952 = (tmp950 * tmp950) + (tmp951 * tmp951);
  tmp953 = 1.5;
  if(tmp952 < 0.0 && tmp953 != 0.0)
  {
    tmp955 = modf(tmp953, &tmp956);
    
    if(tmp955 > 0.5)
    {
      tmp955 -= 1.0;
      tmp956 += 1.0;
    }
    else if(tmp955 < -0.5)
    {
      tmp955 += 1.0;
      tmp956 -= 1.0;
    }
    
    if(fabs(tmp955) < 1e-10)
      tmp954 = pow(tmp952, tmp956);
    else
    {
      tmp958 = modf(1.0/tmp953, &tmp957);
      if(tmp958 > 0.5)
      {
        tmp958 -= 1.0;
        tmp957 += 1.0;
      }
      else if(tmp958 < -0.5)
      {
        tmp958 += 1.0;
        tmp957 -= 1.0;
      }
      if(fabs(tmp958) < 1e-10 && ((unsigned long)tmp957 & 1))
      {
        tmp954 = -pow(-tmp952, tmp955)*pow(tmp952, tmp956);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp952, tmp953);
      }
    }
  }
  else
  {
    tmp954 = pow(tmp952, tmp953);
  }
  if(isnan(tmp954) || isinf(tmp954))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp952, tmp953);
  }tmp959 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp954,"(r_init[405] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp959 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[405] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp959);
    }
  }
  (data->simulationInfo->realParameter[909] /* omega_c[405] PARAM */) = sqrt(tmp959);
  TRACE_POP
}

/*
equation index: 13193
type: SIMPLE_ASSIGN
r_init[404] = r_min + 404.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13193};
  (data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (404.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13194
type: SIMPLE_ASSIGN
omega_c[404] = sqrt(G * Md / (r_init[404] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13194(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13194};
  modelica_real tmp960;
  modelica_real tmp961;
  modelica_real tmp962;
  modelica_real tmp963;
  modelica_real tmp964;
  modelica_real tmp965;
  modelica_real tmp966;
  modelica_real tmp967;
  modelica_real tmp968;
  modelica_real tmp969;
  tmp960 = (data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */);
  tmp961 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp962 = (tmp960 * tmp960) + (tmp961 * tmp961);
  tmp963 = 1.5;
  if(tmp962 < 0.0 && tmp963 != 0.0)
  {
    tmp965 = modf(tmp963, &tmp966);
    
    if(tmp965 > 0.5)
    {
      tmp965 -= 1.0;
      tmp966 += 1.0;
    }
    else if(tmp965 < -0.5)
    {
      tmp965 += 1.0;
      tmp966 -= 1.0;
    }
    
    if(fabs(tmp965) < 1e-10)
      tmp964 = pow(tmp962, tmp966);
    else
    {
      tmp968 = modf(1.0/tmp963, &tmp967);
      if(tmp968 > 0.5)
      {
        tmp968 -= 1.0;
        tmp967 += 1.0;
      }
      else if(tmp968 < -0.5)
      {
        tmp968 += 1.0;
        tmp967 -= 1.0;
      }
      if(fabs(tmp968) < 1e-10 && ((unsigned long)tmp967 & 1))
      {
        tmp964 = -pow(-tmp962, tmp965)*pow(tmp962, tmp966);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp962, tmp963);
      }
    }
  }
  else
  {
    tmp964 = pow(tmp962, tmp963);
  }
  if(isnan(tmp964) || isinf(tmp964))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp962, tmp963);
  }tmp969 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp964,"(r_init[404] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp969 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[404] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp969);
    }
  }
  (data->simulationInfo->realParameter[908] /* omega_c[404] PARAM */) = sqrt(tmp969);
  TRACE_POP
}

/*
equation index: 13195
type: SIMPLE_ASSIGN
r_init[403] = r_min + 403.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13195};
  (data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (403.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13196
type: SIMPLE_ASSIGN
omega_c[403] = sqrt(G * Md / (r_init[403] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13196};
  modelica_real tmp970;
  modelica_real tmp971;
  modelica_real tmp972;
  modelica_real tmp973;
  modelica_real tmp974;
  modelica_real tmp975;
  modelica_real tmp976;
  modelica_real tmp977;
  modelica_real tmp978;
  modelica_real tmp979;
  tmp970 = (data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */);
  tmp971 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp972 = (tmp970 * tmp970) + (tmp971 * tmp971);
  tmp973 = 1.5;
  if(tmp972 < 0.0 && tmp973 != 0.0)
  {
    tmp975 = modf(tmp973, &tmp976);
    
    if(tmp975 > 0.5)
    {
      tmp975 -= 1.0;
      tmp976 += 1.0;
    }
    else if(tmp975 < -0.5)
    {
      tmp975 += 1.0;
      tmp976 -= 1.0;
    }
    
    if(fabs(tmp975) < 1e-10)
      tmp974 = pow(tmp972, tmp976);
    else
    {
      tmp978 = modf(1.0/tmp973, &tmp977);
      if(tmp978 > 0.5)
      {
        tmp978 -= 1.0;
        tmp977 += 1.0;
      }
      else if(tmp978 < -0.5)
      {
        tmp978 += 1.0;
        tmp977 -= 1.0;
      }
      if(fabs(tmp978) < 1e-10 && ((unsigned long)tmp977 & 1))
      {
        tmp974 = -pow(-tmp972, tmp975)*pow(tmp972, tmp976);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp972, tmp973);
      }
    }
  }
  else
  {
    tmp974 = pow(tmp972, tmp973);
  }
  if(isnan(tmp974) || isinf(tmp974))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp972, tmp973);
  }tmp979 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp974,"(r_init[403] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp979 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[403] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp979);
    }
  }
  (data->simulationInfo->realParameter[907] /* omega_c[403] PARAM */) = sqrt(tmp979);
  TRACE_POP
}

/*
equation index: 13197
type: SIMPLE_ASSIGN
r_init[402] = r_min + 402.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13197};
  (data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (402.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13198
type: SIMPLE_ASSIGN
omega_c[402] = sqrt(G * Md / (r_init[402] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13198(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13198};
  modelica_real tmp980;
  modelica_real tmp981;
  modelica_real tmp982;
  modelica_real tmp983;
  modelica_real tmp984;
  modelica_real tmp985;
  modelica_real tmp986;
  modelica_real tmp987;
  modelica_real tmp988;
  modelica_real tmp989;
  tmp980 = (data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */);
  tmp981 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp982 = (tmp980 * tmp980) + (tmp981 * tmp981);
  tmp983 = 1.5;
  if(tmp982 < 0.0 && tmp983 != 0.0)
  {
    tmp985 = modf(tmp983, &tmp986);
    
    if(tmp985 > 0.5)
    {
      tmp985 -= 1.0;
      tmp986 += 1.0;
    }
    else if(tmp985 < -0.5)
    {
      tmp985 += 1.0;
      tmp986 -= 1.0;
    }
    
    if(fabs(tmp985) < 1e-10)
      tmp984 = pow(tmp982, tmp986);
    else
    {
      tmp988 = modf(1.0/tmp983, &tmp987);
      if(tmp988 > 0.5)
      {
        tmp988 -= 1.0;
        tmp987 += 1.0;
      }
      else if(tmp988 < -0.5)
      {
        tmp988 += 1.0;
        tmp987 -= 1.0;
      }
      if(fabs(tmp988) < 1e-10 && ((unsigned long)tmp987 & 1))
      {
        tmp984 = -pow(-tmp982, tmp985)*pow(tmp982, tmp986);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp982, tmp983);
      }
    }
  }
  else
  {
    tmp984 = pow(tmp982, tmp983);
  }
  if(isnan(tmp984) || isinf(tmp984))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp982, tmp983);
  }tmp989 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp984,"(r_init[402] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp989 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[402] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp989);
    }
  }
  (data->simulationInfo->realParameter[906] /* omega_c[402] PARAM */) = sqrt(tmp989);
  TRACE_POP
}

/*
equation index: 13199
type: SIMPLE_ASSIGN
r_init[401] = r_min + 401.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13199};
  (data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (401.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13200
type: SIMPLE_ASSIGN
omega_c[401] = sqrt(G * Md / (r_init[401] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13200(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13200};
  modelica_real tmp990;
  modelica_real tmp991;
  modelica_real tmp992;
  modelica_real tmp993;
  modelica_real tmp994;
  modelica_real tmp995;
  modelica_real tmp996;
  modelica_real tmp997;
  modelica_real tmp998;
  modelica_real tmp999;
  tmp990 = (data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */);
  tmp991 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp992 = (tmp990 * tmp990) + (tmp991 * tmp991);
  tmp993 = 1.5;
  if(tmp992 < 0.0 && tmp993 != 0.0)
  {
    tmp995 = modf(tmp993, &tmp996);
    
    if(tmp995 > 0.5)
    {
      tmp995 -= 1.0;
      tmp996 += 1.0;
    }
    else if(tmp995 < -0.5)
    {
      tmp995 += 1.0;
      tmp996 -= 1.0;
    }
    
    if(fabs(tmp995) < 1e-10)
      tmp994 = pow(tmp992, tmp996);
    else
    {
      tmp998 = modf(1.0/tmp993, &tmp997);
      if(tmp998 > 0.5)
      {
        tmp998 -= 1.0;
        tmp997 += 1.0;
      }
      else if(tmp998 < -0.5)
      {
        tmp998 += 1.0;
        tmp997 -= 1.0;
      }
      if(fabs(tmp998) < 1e-10 && ((unsigned long)tmp997 & 1))
      {
        tmp994 = -pow(-tmp992, tmp995)*pow(tmp992, tmp996);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp992, tmp993);
      }
    }
  }
  else
  {
    tmp994 = pow(tmp992, tmp993);
  }
  if(isnan(tmp994) || isinf(tmp994))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp992, tmp993);
  }tmp999 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp994,"(r_init[401] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp999 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[401] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp999);
    }
  }
  (data->simulationInfo->realParameter[905] /* omega_c[401] PARAM */) = sqrt(tmp999);
  TRACE_POP
}

/*
equation index: 13201
type: SIMPLE_ASSIGN
r_init[400] = r_min + 400.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13201};
  (data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (400.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13202
type: SIMPLE_ASSIGN
omega_c[400] = sqrt(G * Md / (r_init[400] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13202(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13202};
  modelica_real tmp1000;
  modelica_real tmp1001;
  modelica_real tmp1002;
  modelica_real tmp1003;
  modelica_real tmp1004;
  modelica_real tmp1005;
  modelica_real tmp1006;
  modelica_real tmp1007;
  modelica_real tmp1008;
  modelica_real tmp1009;
  tmp1000 = (data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */);
  tmp1001 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1002 = (tmp1000 * tmp1000) + (tmp1001 * tmp1001);
  tmp1003 = 1.5;
  if(tmp1002 < 0.0 && tmp1003 != 0.0)
  {
    tmp1005 = modf(tmp1003, &tmp1006);
    
    if(tmp1005 > 0.5)
    {
      tmp1005 -= 1.0;
      tmp1006 += 1.0;
    }
    else if(tmp1005 < -0.5)
    {
      tmp1005 += 1.0;
      tmp1006 -= 1.0;
    }
    
    if(fabs(tmp1005) < 1e-10)
      tmp1004 = pow(tmp1002, tmp1006);
    else
    {
      tmp1008 = modf(1.0/tmp1003, &tmp1007);
      if(tmp1008 > 0.5)
      {
        tmp1008 -= 1.0;
        tmp1007 += 1.0;
      }
      else if(tmp1008 < -0.5)
      {
        tmp1008 += 1.0;
        tmp1007 -= 1.0;
      }
      if(fabs(tmp1008) < 1e-10 && ((unsigned long)tmp1007 & 1))
      {
        tmp1004 = -pow(-tmp1002, tmp1005)*pow(tmp1002, tmp1006);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1002, tmp1003);
      }
    }
  }
  else
  {
    tmp1004 = pow(tmp1002, tmp1003);
  }
  if(isnan(tmp1004) || isinf(tmp1004))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1002, tmp1003);
  }tmp1009 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1004,"(r_init[400] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1009 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[400] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1009);
    }
  }
  (data->simulationInfo->realParameter[904] /* omega_c[400] PARAM */) = sqrt(tmp1009);
  TRACE_POP
}

/*
equation index: 13203
type: SIMPLE_ASSIGN
r_init[399] = r_min + 399.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13203};
  (data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (399.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13204
type: SIMPLE_ASSIGN
omega_c[399] = sqrt(G * Md / (r_init[399] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13204};
  modelica_real tmp1010;
  modelica_real tmp1011;
  modelica_real tmp1012;
  modelica_real tmp1013;
  modelica_real tmp1014;
  modelica_real tmp1015;
  modelica_real tmp1016;
  modelica_real tmp1017;
  modelica_real tmp1018;
  modelica_real tmp1019;
  tmp1010 = (data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */);
  tmp1011 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1012 = (tmp1010 * tmp1010) + (tmp1011 * tmp1011);
  tmp1013 = 1.5;
  if(tmp1012 < 0.0 && tmp1013 != 0.0)
  {
    tmp1015 = modf(tmp1013, &tmp1016);
    
    if(tmp1015 > 0.5)
    {
      tmp1015 -= 1.0;
      tmp1016 += 1.0;
    }
    else if(tmp1015 < -0.5)
    {
      tmp1015 += 1.0;
      tmp1016 -= 1.0;
    }
    
    if(fabs(tmp1015) < 1e-10)
      tmp1014 = pow(tmp1012, tmp1016);
    else
    {
      tmp1018 = modf(1.0/tmp1013, &tmp1017);
      if(tmp1018 > 0.5)
      {
        tmp1018 -= 1.0;
        tmp1017 += 1.0;
      }
      else if(tmp1018 < -0.5)
      {
        tmp1018 += 1.0;
        tmp1017 -= 1.0;
      }
      if(fabs(tmp1018) < 1e-10 && ((unsigned long)tmp1017 & 1))
      {
        tmp1014 = -pow(-tmp1012, tmp1015)*pow(tmp1012, tmp1016);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1012, tmp1013);
      }
    }
  }
  else
  {
    tmp1014 = pow(tmp1012, tmp1013);
  }
  if(isnan(tmp1014) || isinf(tmp1014))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1012, tmp1013);
  }tmp1019 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1014,"(r_init[399] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1019 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[399] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1019);
    }
  }
  (data->simulationInfo->realParameter[903] /* omega_c[399] PARAM */) = sqrt(tmp1019);
  TRACE_POP
}

/*
equation index: 13205
type: SIMPLE_ASSIGN
r_init[398] = r_min + 398.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13205};
  (data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (398.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13206
type: SIMPLE_ASSIGN
omega_c[398] = sqrt(G * Md / (r_init[398] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13206(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13206};
  modelica_real tmp1020;
  modelica_real tmp1021;
  modelica_real tmp1022;
  modelica_real tmp1023;
  modelica_real tmp1024;
  modelica_real tmp1025;
  modelica_real tmp1026;
  modelica_real tmp1027;
  modelica_real tmp1028;
  modelica_real tmp1029;
  tmp1020 = (data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */);
  tmp1021 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1022 = (tmp1020 * tmp1020) + (tmp1021 * tmp1021);
  tmp1023 = 1.5;
  if(tmp1022 < 0.0 && tmp1023 != 0.0)
  {
    tmp1025 = modf(tmp1023, &tmp1026);
    
    if(tmp1025 > 0.5)
    {
      tmp1025 -= 1.0;
      tmp1026 += 1.0;
    }
    else if(tmp1025 < -0.5)
    {
      tmp1025 += 1.0;
      tmp1026 -= 1.0;
    }
    
    if(fabs(tmp1025) < 1e-10)
      tmp1024 = pow(tmp1022, tmp1026);
    else
    {
      tmp1028 = modf(1.0/tmp1023, &tmp1027);
      if(tmp1028 > 0.5)
      {
        tmp1028 -= 1.0;
        tmp1027 += 1.0;
      }
      else if(tmp1028 < -0.5)
      {
        tmp1028 += 1.0;
        tmp1027 -= 1.0;
      }
      if(fabs(tmp1028) < 1e-10 && ((unsigned long)tmp1027 & 1))
      {
        tmp1024 = -pow(-tmp1022, tmp1025)*pow(tmp1022, tmp1026);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1022, tmp1023);
      }
    }
  }
  else
  {
    tmp1024 = pow(tmp1022, tmp1023);
  }
  if(isnan(tmp1024) || isinf(tmp1024))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1022, tmp1023);
  }tmp1029 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1024,"(r_init[398] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1029 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[398] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1029);
    }
  }
  (data->simulationInfo->realParameter[902] /* omega_c[398] PARAM */) = sqrt(tmp1029);
  TRACE_POP
}

/*
equation index: 13207
type: SIMPLE_ASSIGN
r_init[397] = r_min + 397.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13207};
  (data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (397.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13208
type: SIMPLE_ASSIGN
omega_c[397] = sqrt(G * Md / (r_init[397] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13208(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13208};
  modelica_real tmp1030;
  modelica_real tmp1031;
  modelica_real tmp1032;
  modelica_real tmp1033;
  modelica_real tmp1034;
  modelica_real tmp1035;
  modelica_real tmp1036;
  modelica_real tmp1037;
  modelica_real tmp1038;
  modelica_real tmp1039;
  tmp1030 = (data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */);
  tmp1031 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1032 = (tmp1030 * tmp1030) + (tmp1031 * tmp1031);
  tmp1033 = 1.5;
  if(tmp1032 < 0.0 && tmp1033 != 0.0)
  {
    tmp1035 = modf(tmp1033, &tmp1036);
    
    if(tmp1035 > 0.5)
    {
      tmp1035 -= 1.0;
      tmp1036 += 1.0;
    }
    else if(tmp1035 < -0.5)
    {
      tmp1035 += 1.0;
      tmp1036 -= 1.0;
    }
    
    if(fabs(tmp1035) < 1e-10)
      tmp1034 = pow(tmp1032, tmp1036);
    else
    {
      tmp1038 = modf(1.0/tmp1033, &tmp1037);
      if(tmp1038 > 0.5)
      {
        tmp1038 -= 1.0;
        tmp1037 += 1.0;
      }
      else if(tmp1038 < -0.5)
      {
        tmp1038 += 1.0;
        tmp1037 -= 1.0;
      }
      if(fabs(tmp1038) < 1e-10 && ((unsigned long)tmp1037 & 1))
      {
        tmp1034 = -pow(-tmp1032, tmp1035)*pow(tmp1032, tmp1036);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1032, tmp1033);
      }
    }
  }
  else
  {
    tmp1034 = pow(tmp1032, tmp1033);
  }
  if(isnan(tmp1034) || isinf(tmp1034))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1032, tmp1033);
  }tmp1039 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1034,"(r_init[397] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1039 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[397] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1039);
    }
  }
  (data->simulationInfo->realParameter[901] /* omega_c[397] PARAM */) = sqrt(tmp1039);
  TRACE_POP
}

/*
equation index: 13209
type: SIMPLE_ASSIGN
r_init[396] = r_min + 396.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13209};
  (data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (396.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13210
type: SIMPLE_ASSIGN
omega_c[396] = sqrt(G * Md / (r_init[396] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13210(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13210};
  modelica_real tmp1040;
  modelica_real tmp1041;
  modelica_real tmp1042;
  modelica_real tmp1043;
  modelica_real tmp1044;
  modelica_real tmp1045;
  modelica_real tmp1046;
  modelica_real tmp1047;
  modelica_real tmp1048;
  modelica_real tmp1049;
  tmp1040 = (data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */);
  tmp1041 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1042 = (tmp1040 * tmp1040) + (tmp1041 * tmp1041);
  tmp1043 = 1.5;
  if(tmp1042 < 0.0 && tmp1043 != 0.0)
  {
    tmp1045 = modf(tmp1043, &tmp1046);
    
    if(tmp1045 > 0.5)
    {
      tmp1045 -= 1.0;
      tmp1046 += 1.0;
    }
    else if(tmp1045 < -0.5)
    {
      tmp1045 += 1.0;
      tmp1046 -= 1.0;
    }
    
    if(fabs(tmp1045) < 1e-10)
      tmp1044 = pow(tmp1042, tmp1046);
    else
    {
      tmp1048 = modf(1.0/tmp1043, &tmp1047);
      if(tmp1048 > 0.5)
      {
        tmp1048 -= 1.0;
        tmp1047 += 1.0;
      }
      else if(tmp1048 < -0.5)
      {
        tmp1048 += 1.0;
        tmp1047 -= 1.0;
      }
      if(fabs(tmp1048) < 1e-10 && ((unsigned long)tmp1047 & 1))
      {
        tmp1044 = -pow(-tmp1042, tmp1045)*pow(tmp1042, tmp1046);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1042, tmp1043);
      }
    }
  }
  else
  {
    tmp1044 = pow(tmp1042, tmp1043);
  }
  if(isnan(tmp1044) || isinf(tmp1044))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1042, tmp1043);
  }tmp1049 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1044,"(r_init[396] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1049 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[396] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1049);
    }
  }
  (data->simulationInfo->realParameter[900] /* omega_c[396] PARAM */) = sqrt(tmp1049);
  TRACE_POP
}

/*
equation index: 13211
type: SIMPLE_ASSIGN
r_init[395] = r_min + 395.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13211};
  (data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (395.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13212
type: SIMPLE_ASSIGN
omega_c[395] = sqrt(G * Md / (r_init[395] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13212};
  modelica_real tmp1050;
  modelica_real tmp1051;
  modelica_real tmp1052;
  modelica_real tmp1053;
  modelica_real tmp1054;
  modelica_real tmp1055;
  modelica_real tmp1056;
  modelica_real tmp1057;
  modelica_real tmp1058;
  modelica_real tmp1059;
  tmp1050 = (data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */);
  tmp1051 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1052 = (tmp1050 * tmp1050) + (tmp1051 * tmp1051);
  tmp1053 = 1.5;
  if(tmp1052 < 0.0 && tmp1053 != 0.0)
  {
    tmp1055 = modf(tmp1053, &tmp1056);
    
    if(tmp1055 > 0.5)
    {
      tmp1055 -= 1.0;
      tmp1056 += 1.0;
    }
    else if(tmp1055 < -0.5)
    {
      tmp1055 += 1.0;
      tmp1056 -= 1.0;
    }
    
    if(fabs(tmp1055) < 1e-10)
      tmp1054 = pow(tmp1052, tmp1056);
    else
    {
      tmp1058 = modf(1.0/tmp1053, &tmp1057);
      if(tmp1058 > 0.5)
      {
        tmp1058 -= 1.0;
        tmp1057 += 1.0;
      }
      else if(tmp1058 < -0.5)
      {
        tmp1058 += 1.0;
        tmp1057 -= 1.0;
      }
      if(fabs(tmp1058) < 1e-10 && ((unsigned long)tmp1057 & 1))
      {
        tmp1054 = -pow(-tmp1052, tmp1055)*pow(tmp1052, tmp1056);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1052, tmp1053);
      }
    }
  }
  else
  {
    tmp1054 = pow(tmp1052, tmp1053);
  }
  if(isnan(tmp1054) || isinf(tmp1054))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1052, tmp1053);
  }tmp1059 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1054,"(r_init[395] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1059 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[395] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1059);
    }
  }
  (data->simulationInfo->realParameter[899] /* omega_c[395] PARAM */) = sqrt(tmp1059);
  TRACE_POP
}

/*
equation index: 13213
type: SIMPLE_ASSIGN
r_init[394] = r_min + 394.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13213};
  (data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (394.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13214
type: SIMPLE_ASSIGN
omega_c[394] = sqrt(G * Md / (r_init[394] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13214(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13214};
  modelica_real tmp1060;
  modelica_real tmp1061;
  modelica_real tmp1062;
  modelica_real tmp1063;
  modelica_real tmp1064;
  modelica_real tmp1065;
  modelica_real tmp1066;
  modelica_real tmp1067;
  modelica_real tmp1068;
  modelica_real tmp1069;
  tmp1060 = (data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */);
  tmp1061 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1062 = (tmp1060 * tmp1060) + (tmp1061 * tmp1061);
  tmp1063 = 1.5;
  if(tmp1062 < 0.0 && tmp1063 != 0.0)
  {
    tmp1065 = modf(tmp1063, &tmp1066);
    
    if(tmp1065 > 0.5)
    {
      tmp1065 -= 1.0;
      tmp1066 += 1.0;
    }
    else if(tmp1065 < -0.5)
    {
      tmp1065 += 1.0;
      tmp1066 -= 1.0;
    }
    
    if(fabs(tmp1065) < 1e-10)
      tmp1064 = pow(tmp1062, tmp1066);
    else
    {
      tmp1068 = modf(1.0/tmp1063, &tmp1067);
      if(tmp1068 > 0.5)
      {
        tmp1068 -= 1.0;
        tmp1067 += 1.0;
      }
      else if(tmp1068 < -0.5)
      {
        tmp1068 += 1.0;
        tmp1067 -= 1.0;
      }
      if(fabs(tmp1068) < 1e-10 && ((unsigned long)tmp1067 & 1))
      {
        tmp1064 = -pow(-tmp1062, tmp1065)*pow(tmp1062, tmp1066);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1062, tmp1063);
      }
    }
  }
  else
  {
    tmp1064 = pow(tmp1062, tmp1063);
  }
  if(isnan(tmp1064) || isinf(tmp1064))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1062, tmp1063);
  }tmp1069 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1064,"(r_init[394] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1069 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[394] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1069);
    }
  }
  (data->simulationInfo->realParameter[898] /* omega_c[394] PARAM */) = sqrt(tmp1069);
  TRACE_POP
}

/*
equation index: 13215
type: SIMPLE_ASSIGN
r_init[393] = r_min + 393.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13215};
  (data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (393.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13216
type: SIMPLE_ASSIGN
omega_c[393] = sqrt(G * Md / (r_init[393] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13216(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13216};
  modelica_real tmp1070;
  modelica_real tmp1071;
  modelica_real tmp1072;
  modelica_real tmp1073;
  modelica_real tmp1074;
  modelica_real tmp1075;
  modelica_real tmp1076;
  modelica_real tmp1077;
  modelica_real tmp1078;
  modelica_real tmp1079;
  tmp1070 = (data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */);
  tmp1071 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1072 = (tmp1070 * tmp1070) + (tmp1071 * tmp1071);
  tmp1073 = 1.5;
  if(tmp1072 < 0.0 && tmp1073 != 0.0)
  {
    tmp1075 = modf(tmp1073, &tmp1076);
    
    if(tmp1075 > 0.5)
    {
      tmp1075 -= 1.0;
      tmp1076 += 1.0;
    }
    else if(tmp1075 < -0.5)
    {
      tmp1075 += 1.0;
      tmp1076 -= 1.0;
    }
    
    if(fabs(tmp1075) < 1e-10)
      tmp1074 = pow(tmp1072, tmp1076);
    else
    {
      tmp1078 = modf(1.0/tmp1073, &tmp1077);
      if(tmp1078 > 0.5)
      {
        tmp1078 -= 1.0;
        tmp1077 += 1.0;
      }
      else if(tmp1078 < -0.5)
      {
        tmp1078 += 1.0;
        tmp1077 -= 1.0;
      }
      if(fabs(tmp1078) < 1e-10 && ((unsigned long)tmp1077 & 1))
      {
        tmp1074 = -pow(-tmp1072, tmp1075)*pow(tmp1072, tmp1076);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1072, tmp1073);
      }
    }
  }
  else
  {
    tmp1074 = pow(tmp1072, tmp1073);
  }
  if(isnan(tmp1074) || isinf(tmp1074))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1072, tmp1073);
  }tmp1079 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1074,"(r_init[393] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1079 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[393] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1079);
    }
  }
  (data->simulationInfo->realParameter[897] /* omega_c[393] PARAM */) = sqrt(tmp1079);
  TRACE_POP
}

/*
equation index: 13217
type: SIMPLE_ASSIGN
r_init[392] = r_min + 392.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13217};
  (data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (392.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13218
type: SIMPLE_ASSIGN
omega_c[392] = sqrt(G * Md / (r_init[392] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13218(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13218};
  modelica_real tmp1080;
  modelica_real tmp1081;
  modelica_real tmp1082;
  modelica_real tmp1083;
  modelica_real tmp1084;
  modelica_real tmp1085;
  modelica_real tmp1086;
  modelica_real tmp1087;
  modelica_real tmp1088;
  modelica_real tmp1089;
  tmp1080 = (data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */);
  tmp1081 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1082 = (tmp1080 * tmp1080) + (tmp1081 * tmp1081);
  tmp1083 = 1.5;
  if(tmp1082 < 0.0 && tmp1083 != 0.0)
  {
    tmp1085 = modf(tmp1083, &tmp1086);
    
    if(tmp1085 > 0.5)
    {
      tmp1085 -= 1.0;
      tmp1086 += 1.0;
    }
    else if(tmp1085 < -0.5)
    {
      tmp1085 += 1.0;
      tmp1086 -= 1.0;
    }
    
    if(fabs(tmp1085) < 1e-10)
      tmp1084 = pow(tmp1082, tmp1086);
    else
    {
      tmp1088 = modf(1.0/tmp1083, &tmp1087);
      if(tmp1088 > 0.5)
      {
        tmp1088 -= 1.0;
        tmp1087 += 1.0;
      }
      else if(tmp1088 < -0.5)
      {
        tmp1088 += 1.0;
        tmp1087 -= 1.0;
      }
      if(fabs(tmp1088) < 1e-10 && ((unsigned long)tmp1087 & 1))
      {
        tmp1084 = -pow(-tmp1082, tmp1085)*pow(tmp1082, tmp1086);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1082, tmp1083);
      }
    }
  }
  else
  {
    tmp1084 = pow(tmp1082, tmp1083);
  }
  if(isnan(tmp1084) || isinf(tmp1084))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1082, tmp1083);
  }tmp1089 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1084,"(r_init[392] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1089 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[392] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1089);
    }
  }
  (data->simulationInfo->realParameter[896] /* omega_c[392] PARAM */) = sqrt(tmp1089);
  TRACE_POP
}

/*
equation index: 13219
type: SIMPLE_ASSIGN
r_init[391] = r_min + 391.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13219};
  (data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (391.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13220
type: SIMPLE_ASSIGN
omega_c[391] = sqrt(G * Md / (r_init[391] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13220};
  modelica_real tmp1090;
  modelica_real tmp1091;
  modelica_real tmp1092;
  modelica_real tmp1093;
  modelica_real tmp1094;
  modelica_real tmp1095;
  modelica_real tmp1096;
  modelica_real tmp1097;
  modelica_real tmp1098;
  modelica_real tmp1099;
  tmp1090 = (data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */);
  tmp1091 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1092 = (tmp1090 * tmp1090) + (tmp1091 * tmp1091);
  tmp1093 = 1.5;
  if(tmp1092 < 0.0 && tmp1093 != 0.0)
  {
    tmp1095 = modf(tmp1093, &tmp1096);
    
    if(tmp1095 > 0.5)
    {
      tmp1095 -= 1.0;
      tmp1096 += 1.0;
    }
    else if(tmp1095 < -0.5)
    {
      tmp1095 += 1.0;
      tmp1096 -= 1.0;
    }
    
    if(fabs(tmp1095) < 1e-10)
      tmp1094 = pow(tmp1092, tmp1096);
    else
    {
      tmp1098 = modf(1.0/tmp1093, &tmp1097);
      if(tmp1098 > 0.5)
      {
        tmp1098 -= 1.0;
        tmp1097 += 1.0;
      }
      else if(tmp1098 < -0.5)
      {
        tmp1098 += 1.0;
        tmp1097 -= 1.0;
      }
      if(fabs(tmp1098) < 1e-10 && ((unsigned long)tmp1097 & 1))
      {
        tmp1094 = -pow(-tmp1092, tmp1095)*pow(tmp1092, tmp1096);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1092, tmp1093);
      }
    }
  }
  else
  {
    tmp1094 = pow(tmp1092, tmp1093);
  }
  if(isnan(tmp1094) || isinf(tmp1094))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1092, tmp1093);
  }tmp1099 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1094,"(r_init[391] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1099 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[391] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1099);
    }
  }
  (data->simulationInfo->realParameter[895] /* omega_c[391] PARAM */) = sqrt(tmp1099);
  TRACE_POP
}

/*
equation index: 13221
type: SIMPLE_ASSIGN
r_init[390] = r_min + 390.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13221};
  (data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (390.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13222
type: SIMPLE_ASSIGN
omega_c[390] = sqrt(G * Md / (r_init[390] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13222(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13222};
  modelica_real tmp1100;
  modelica_real tmp1101;
  modelica_real tmp1102;
  modelica_real tmp1103;
  modelica_real tmp1104;
  modelica_real tmp1105;
  modelica_real tmp1106;
  modelica_real tmp1107;
  modelica_real tmp1108;
  modelica_real tmp1109;
  tmp1100 = (data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */);
  tmp1101 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1102 = (tmp1100 * tmp1100) + (tmp1101 * tmp1101);
  tmp1103 = 1.5;
  if(tmp1102 < 0.0 && tmp1103 != 0.0)
  {
    tmp1105 = modf(tmp1103, &tmp1106);
    
    if(tmp1105 > 0.5)
    {
      tmp1105 -= 1.0;
      tmp1106 += 1.0;
    }
    else if(tmp1105 < -0.5)
    {
      tmp1105 += 1.0;
      tmp1106 -= 1.0;
    }
    
    if(fabs(tmp1105) < 1e-10)
      tmp1104 = pow(tmp1102, tmp1106);
    else
    {
      tmp1108 = modf(1.0/tmp1103, &tmp1107);
      if(tmp1108 > 0.5)
      {
        tmp1108 -= 1.0;
        tmp1107 += 1.0;
      }
      else if(tmp1108 < -0.5)
      {
        tmp1108 += 1.0;
        tmp1107 -= 1.0;
      }
      if(fabs(tmp1108) < 1e-10 && ((unsigned long)tmp1107 & 1))
      {
        tmp1104 = -pow(-tmp1102, tmp1105)*pow(tmp1102, tmp1106);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1102, tmp1103);
      }
    }
  }
  else
  {
    tmp1104 = pow(tmp1102, tmp1103);
  }
  if(isnan(tmp1104) || isinf(tmp1104))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1102, tmp1103);
  }tmp1109 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1104,"(r_init[390] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1109 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[390] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1109);
    }
  }
  (data->simulationInfo->realParameter[894] /* omega_c[390] PARAM */) = sqrt(tmp1109);
  TRACE_POP
}

/*
equation index: 13223
type: SIMPLE_ASSIGN
r_init[389] = r_min + 389.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13223};
  (data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (389.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13224
type: SIMPLE_ASSIGN
omega_c[389] = sqrt(G * Md / (r_init[389] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13224(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13224};
  modelica_real tmp1110;
  modelica_real tmp1111;
  modelica_real tmp1112;
  modelica_real tmp1113;
  modelica_real tmp1114;
  modelica_real tmp1115;
  modelica_real tmp1116;
  modelica_real tmp1117;
  modelica_real tmp1118;
  modelica_real tmp1119;
  tmp1110 = (data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */);
  tmp1111 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1112 = (tmp1110 * tmp1110) + (tmp1111 * tmp1111);
  tmp1113 = 1.5;
  if(tmp1112 < 0.0 && tmp1113 != 0.0)
  {
    tmp1115 = modf(tmp1113, &tmp1116);
    
    if(tmp1115 > 0.5)
    {
      tmp1115 -= 1.0;
      tmp1116 += 1.0;
    }
    else if(tmp1115 < -0.5)
    {
      tmp1115 += 1.0;
      tmp1116 -= 1.0;
    }
    
    if(fabs(tmp1115) < 1e-10)
      tmp1114 = pow(tmp1112, tmp1116);
    else
    {
      tmp1118 = modf(1.0/tmp1113, &tmp1117);
      if(tmp1118 > 0.5)
      {
        tmp1118 -= 1.0;
        tmp1117 += 1.0;
      }
      else if(tmp1118 < -0.5)
      {
        tmp1118 += 1.0;
        tmp1117 -= 1.0;
      }
      if(fabs(tmp1118) < 1e-10 && ((unsigned long)tmp1117 & 1))
      {
        tmp1114 = -pow(-tmp1112, tmp1115)*pow(tmp1112, tmp1116);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1112, tmp1113);
      }
    }
  }
  else
  {
    tmp1114 = pow(tmp1112, tmp1113);
  }
  if(isnan(tmp1114) || isinf(tmp1114))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1112, tmp1113);
  }tmp1119 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1114,"(r_init[389] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1119 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[389] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1119);
    }
  }
  (data->simulationInfo->realParameter[893] /* omega_c[389] PARAM */) = sqrt(tmp1119);
  TRACE_POP
}

/*
equation index: 13225
type: SIMPLE_ASSIGN
r_init[388] = r_min + 388.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13225};
  (data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (388.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13226
type: SIMPLE_ASSIGN
omega_c[388] = sqrt(G * Md / (r_init[388] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13226(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13226};
  modelica_real tmp1120;
  modelica_real tmp1121;
  modelica_real tmp1122;
  modelica_real tmp1123;
  modelica_real tmp1124;
  modelica_real tmp1125;
  modelica_real tmp1126;
  modelica_real tmp1127;
  modelica_real tmp1128;
  modelica_real tmp1129;
  tmp1120 = (data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */);
  tmp1121 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1122 = (tmp1120 * tmp1120) + (tmp1121 * tmp1121);
  tmp1123 = 1.5;
  if(tmp1122 < 0.0 && tmp1123 != 0.0)
  {
    tmp1125 = modf(tmp1123, &tmp1126);
    
    if(tmp1125 > 0.5)
    {
      tmp1125 -= 1.0;
      tmp1126 += 1.0;
    }
    else if(tmp1125 < -0.5)
    {
      tmp1125 += 1.0;
      tmp1126 -= 1.0;
    }
    
    if(fabs(tmp1125) < 1e-10)
      tmp1124 = pow(tmp1122, tmp1126);
    else
    {
      tmp1128 = modf(1.0/tmp1123, &tmp1127);
      if(tmp1128 > 0.5)
      {
        tmp1128 -= 1.0;
        tmp1127 += 1.0;
      }
      else if(tmp1128 < -0.5)
      {
        tmp1128 += 1.0;
        tmp1127 -= 1.0;
      }
      if(fabs(tmp1128) < 1e-10 && ((unsigned long)tmp1127 & 1))
      {
        tmp1124 = -pow(-tmp1122, tmp1125)*pow(tmp1122, tmp1126);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1122, tmp1123);
      }
    }
  }
  else
  {
    tmp1124 = pow(tmp1122, tmp1123);
  }
  if(isnan(tmp1124) || isinf(tmp1124))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1122, tmp1123);
  }tmp1129 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1124,"(r_init[388] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1129 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[388] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1129);
    }
  }
  (data->simulationInfo->realParameter[892] /* omega_c[388] PARAM */) = sqrt(tmp1129);
  TRACE_POP
}

/*
equation index: 13227
type: SIMPLE_ASSIGN
r_init[387] = r_min + 387.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13227};
  (data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (387.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13228
type: SIMPLE_ASSIGN
omega_c[387] = sqrt(G * Md / (r_init[387] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13228};
  modelica_real tmp1130;
  modelica_real tmp1131;
  modelica_real tmp1132;
  modelica_real tmp1133;
  modelica_real tmp1134;
  modelica_real tmp1135;
  modelica_real tmp1136;
  modelica_real tmp1137;
  modelica_real tmp1138;
  modelica_real tmp1139;
  tmp1130 = (data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */);
  tmp1131 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1132 = (tmp1130 * tmp1130) + (tmp1131 * tmp1131);
  tmp1133 = 1.5;
  if(tmp1132 < 0.0 && tmp1133 != 0.0)
  {
    tmp1135 = modf(tmp1133, &tmp1136);
    
    if(tmp1135 > 0.5)
    {
      tmp1135 -= 1.0;
      tmp1136 += 1.0;
    }
    else if(tmp1135 < -0.5)
    {
      tmp1135 += 1.0;
      tmp1136 -= 1.0;
    }
    
    if(fabs(tmp1135) < 1e-10)
      tmp1134 = pow(tmp1132, tmp1136);
    else
    {
      tmp1138 = modf(1.0/tmp1133, &tmp1137);
      if(tmp1138 > 0.5)
      {
        tmp1138 -= 1.0;
        tmp1137 += 1.0;
      }
      else if(tmp1138 < -0.5)
      {
        tmp1138 += 1.0;
        tmp1137 -= 1.0;
      }
      if(fabs(tmp1138) < 1e-10 && ((unsigned long)tmp1137 & 1))
      {
        tmp1134 = -pow(-tmp1132, tmp1135)*pow(tmp1132, tmp1136);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1132, tmp1133);
      }
    }
  }
  else
  {
    tmp1134 = pow(tmp1132, tmp1133);
  }
  if(isnan(tmp1134) || isinf(tmp1134))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1132, tmp1133);
  }tmp1139 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1134,"(r_init[387] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1139 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[387] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1139);
    }
  }
  (data->simulationInfo->realParameter[891] /* omega_c[387] PARAM */) = sqrt(tmp1139);
  TRACE_POP
}

/*
equation index: 13229
type: SIMPLE_ASSIGN
r_init[386] = r_min + 386.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13229};
  (data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (386.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13230
type: SIMPLE_ASSIGN
omega_c[386] = sqrt(G * Md / (r_init[386] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13230(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13230};
  modelica_real tmp1140;
  modelica_real tmp1141;
  modelica_real tmp1142;
  modelica_real tmp1143;
  modelica_real tmp1144;
  modelica_real tmp1145;
  modelica_real tmp1146;
  modelica_real tmp1147;
  modelica_real tmp1148;
  modelica_real tmp1149;
  tmp1140 = (data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */);
  tmp1141 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1142 = (tmp1140 * tmp1140) + (tmp1141 * tmp1141);
  tmp1143 = 1.5;
  if(tmp1142 < 0.0 && tmp1143 != 0.0)
  {
    tmp1145 = modf(tmp1143, &tmp1146);
    
    if(tmp1145 > 0.5)
    {
      tmp1145 -= 1.0;
      tmp1146 += 1.0;
    }
    else if(tmp1145 < -0.5)
    {
      tmp1145 += 1.0;
      tmp1146 -= 1.0;
    }
    
    if(fabs(tmp1145) < 1e-10)
      tmp1144 = pow(tmp1142, tmp1146);
    else
    {
      tmp1148 = modf(1.0/tmp1143, &tmp1147);
      if(tmp1148 > 0.5)
      {
        tmp1148 -= 1.0;
        tmp1147 += 1.0;
      }
      else if(tmp1148 < -0.5)
      {
        tmp1148 += 1.0;
        tmp1147 -= 1.0;
      }
      if(fabs(tmp1148) < 1e-10 && ((unsigned long)tmp1147 & 1))
      {
        tmp1144 = -pow(-tmp1142, tmp1145)*pow(tmp1142, tmp1146);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1142, tmp1143);
      }
    }
  }
  else
  {
    tmp1144 = pow(tmp1142, tmp1143);
  }
  if(isnan(tmp1144) || isinf(tmp1144))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1142, tmp1143);
  }tmp1149 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1144,"(r_init[386] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1149 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[386] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1149);
    }
  }
  (data->simulationInfo->realParameter[890] /* omega_c[386] PARAM */) = sqrt(tmp1149);
  TRACE_POP
}

/*
equation index: 13231
type: SIMPLE_ASSIGN
r_init[385] = r_min + 385.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13231};
  (data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (385.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13232
type: SIMPLE_ASSIGN
omega_c[385] = sqrt(G * Md / (r_init[385] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13232(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13232};
  modelica_real tmp1150;
  modelica_real tmp1151;
  modelica_real tmp1152;
  modelica_real tmp1153;
  modelica_real tmp1154;
  modelica_real tmp1155;
  modelica_real tmp1156;
  modelica_real tmp1157;
  modelica_real tmp1158;
  modelica_real tmp1159;
  tmp1150 = (data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */);
  tmp1151 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1152 = (tmp1150 * tmp1150) + (tmp1151 * tmp1151);
  tmp1153 = 1.5;
  if(tmp1152 < 0.0 && tmp1153 != 0.0)
  {
    tmp1155 = modf(tmp1153, &tmp1156);
    
    if(tmp1155 > 0.5)
    {
      tmp1155 -= 1.0;
      tmp1156 += 1.0;
    }
    else if(tmp1155 < -0.5)
    {
      tmp1155 += 1.0;
      tmp1156 -= 1.0;
    }
    
    if(fabs(tmp1155) < 1e-10)
      tmp1154 = pow(tmp1152, tmp1156);
    else
    {
      tmp1158 = modf(1.0/tmp1153, &tmp1157);
      if(tmp1158 > 0.5)
      {
        tmp1158 -= 1.0;
        tmp1157 += 1.0;
      }
      else if(tmp1158 < -0.5)
      {
        tmp1158 += 1.0;
        tmp1157 -= 1.0;
      }
      if(fabs(tmp1158) < 1e-10 && ((unsigned long)tmp1157 & 1))
      {
        tmp1154 = -pow(-tmp1152, tmp1155)*pow(tmp1152, tmp1156);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1152, tmp1153);
      }
    }
  }
  else
  {
    tmp1154 = pow(tmp1152, tmp1153);
  }
  if(isnan(tmp1154) || isinf(tmp1154))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1152, tmp1153);
  }tmp1159 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1154,"(r_init[385] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1159 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[385] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1159);
    }
  }
  (data->simulationInfo->realParameter[889] /* omega_c[385] PARAM */) = sqrt(tmp1159);
  TRACE_POP
}

/*
equation index: 13233
type: SIMPLE_ASSIGN
r_init[384] = r_min + 384.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13233};
  (data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (384.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13234
type: SIMPLE_ASSIGN
omega_c[384] = sqrt(G * Md / (r_init[384] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13234(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13234};
  modelica_real tmp1160;
  modelica_real tmp1161;
  modelica_real tmp1162;
  modelica_real tmp1163;
  modelica_real tmp1164;
  modelica_real tmp1165;
  modelica_real tmp1166;
  modelica_real tmp1167;
  modelica_real tmp1168;
  modelica_real tmp1169;
  tmp1160 = (data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */);
  tmp1161 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1162 = (tmp1160 * tmp1160) + (tmp1161 * tmp1161);
  tmp1163 = 1.5;
  if(tmp1162 < 0.0 && tmp1163 != 0.0)
  {
    tmp1165 = modf(tmp1163, &tmp1166);
    
    if(tmp1165 > 0.5)
    {
      tmp1165 -= 1.0;
      tmp1166 += 1.0;
    }
    else if(tmp1165 < -0.5)
    {
      tmp1165 += 1.0;
      tmp1166 -= 1.0;
    }
    
    if(fabs(tmp1165) < 1e-10)
      tmp1164 = pow(tmp1162, tmp1166);
    else
    {
      tmp1168 = modf(1.0/tmp1163, &tmp1167);
      if(tmp1168 > 0.5)
      {
        tmp1168 -= 1.0;
        tmp1167 += 1.0;
      }
      else if(tmp1168 < -0.5)
      {
        tmp1168 += 1.0;
        tmp1167 -= 1.0;
      }
      if(fabs(tmp1168) < 1e-10 && ((unsigned long)tmp1167 & 1))
      {
        tmp1164 = -pow(-tmp1162, tmp1165)*pow(tmp1162, tmp1166);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1162, tmp1163);
      }
    }
  }
  else
  {
    tmp1164 = pow(tmp1162, tmp1163);
  }
  if(isnan(tmp1164) || isinf(tmp1164))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1162, tmp1163);
  }tmp1169 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1164,"(r_init[384] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1169 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[384] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1169);
    }
  }
  (data->simulationInfo->realParameter[888] /* omega_c[384] PARAM */) = sqrt(tmp1169);
  TRACE_POP
}

/*
equation index: 13235
type: SIMPLE_ASSIGN
r_init[383] = r_min + 383.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13235};
  (data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (383.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13236
type: SIMPLE_ASSIGN
omega_c[383] = sqrt(G * Md / (r_init[383] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13236};
  modelica_real tmp1170;
  modelica_real tmp1171;
  modelica_real tmp1172;
  modelica_real tmp1173;
  modelica_real tmp1174;
  modelica_real tmp1175;
  modelica_real tmp1176;
  modelica_real tmp1177;
  modelica_real tmp1178;
  modelica_real tmp1179;
  tmp1170 = (data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */);
  tmp1171 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1172 = (tmp1170 * tmp1170) + (tmp1171 * tmp1171);
  tmp1173 = 1.5;
  if(tmp1172 < 0.0 && tmp1173 != 0.0)
  {
    tmp1175 = modf(tmp1173, &tmp1176);
    
    if(tmp1175 > 0.5)
    {
      tmp1175 -= 1.0;
      tmp1176 += 1.0;
    }
    else if(tmp1175 < -0.5)
    {
      tmp1175 += 1.0;
      tmp1176 -= 1.0;
    }
    
    if(fabs(tmp1175) < 1e-10)
      tmp1174 = pow(tmp1172, tmp1176);
    else
    {
      tmp1178 = modf(1.0/tmp1173, &tmp1177);
      if(tmp1178 > 0.5)
      {
        tmp1178 -= 1.0;
        tmp1177 += 1.0;
      }
      else if(tmp1178 < -0.5)
      {
        tmp1178 += 1.0;
        tmp1177 -= 1.0;
      }
      if(fabs(tmp1178) < 1e-10 && ((unsigned long)tmp1177 & 1))
      {
        tmp1174 = -pow(-tmp1172, tmp1175)*pow(tmp1172, tmp1176);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1172, tmp1173);
      }
    }
  }
  else
  {
    tmp1174 = pow(tmp1172, tmp1173);
  }
  if(isnan(tmp1174) || isinf(tmp1174))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1172, tmp1173);
  }tmp1179 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1174,"(r_init[383] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1179 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[383] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1179);
    }
  }
  (data->simulationInfo->realParameter[887] /* omega_c[383] PARAM */) = sqrt(tmp1179);
  TRACE_POP
}

/*
equation index: 13237
type: SIMPLE_ASSIGN
r_init[382] = r_min + 382.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13237};
  (data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (382.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13238
type: SIMPLE_ASSIGN
omega_c[382] = sqrt(G * Md / (r_init[382] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13238(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13238};
  modelica_real tmp1180;
  modelica_real tmp1181;
  modelica_real tmp1182;
  modelica_real tmp1183;
  modelica_real tmp1184;
  modelica_real tmp1185;
  modelica_real tmp1186;
  modelica_real tmp1187;
  modelica_real tmp1188;
  modelica_real tmp1189;
  tmp1180 = (data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */);
  tmp1181 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1182 = (tmp1180 * tmp1180) + (tmp1181 * tmp1181);
  tmp1183 = 1.5;
  if(tmp1182 < 0.0 && tmp1183 != 0.0)
  {
    tmp1185 = modf(tmp1183, &tmp1186);
    
    if(tmp1185 > 0.5)
    {
      tmp1185 -= 1.0;
      tmp1186 += 1.0;
    }
    else if(tmp1185 < -0.5)
    {
      tmp1185 += 1.0;
      tmp1186 -= 1.0;
    }
    
    if(fabs(tmp1185) < 1e-10)
      tmp1184 = pow(tmp1182, tmp1186);
    else
    {
      tmp1188 = modf(1.0/tmp1183, &tmp1187);
      if(tmp1188 > 0.5)
      {
        tmp1188 -= 1.0;
        tmp1187 += 1.0;
      }
      else if(tmp1188 < -0.5)
      {
        tmp1188 += 1.0;
        tmp1187 -= 1.0;
      }
      if(fabs(tmp1188) < 1e-10 && ((unsigned long)tmp1187 & 1))
      {
        tmp1184 = -pow(-tmp1182, tmp1185)*pow(tmp1182, tmp1186);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1182, tmp1183);
      }
    }
  }
  else
  {
    tmp1184 = pow(tmp1182, tmp1183);
  }
  if(isnan(tmp1184) || isinf(tmp1184))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1182, tmp1183);
  }tmp1189 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1184,"(r_init[382] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1189 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[382] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1189);
    }
  }
  (data->simulationInfo->realParameter[886] /* omega_c[382] PARAM */) = sqrt(tmp1189);
  TRACE_POP
}

/*
equation index: 13239
type: SIMPLE_ASSIGN
r_init[381] = r_min + 381.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13239};
  (data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (381.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13240
type: SIMPLE_ASSIGN
omega_c[381] = sqrt(G * Md / (r_init[381] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13240(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13240};
  modelica_real tmp1190;
  modelica_real tmp1191;
  modelica_real tmp1192;
  modelica_real tmp1193;
  modelica_real tmp1194;
  modelica_real tmp1195;
  modelica_real tmp1196;
  modelica_real tmp1197;
  modelica_real tmp1198;
  modelica_real tmp1199;
  tmp1190 = (data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */);
  tmp1191 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1192 = (tmp1190 * tmp1190) + (tmp1191 * tmp1191);
  tmp1193 = 1.5;
  if(tmp1192 < 0.0 && tmp1193 != 0.0)
  {
    tmp1195 = modf(tmp1193, &tmp1196);
    
    if(tmp1195 > 0.5)
    {
      tmp1195 -= 1.0;
      tmp1196 += 1.0;
    }
    else if(tmp1195 < -0.5)
    {
      tmp1195 += 1.0;
      tmp1196 -= 1.0;
    }
    
    if(fabs(tmp1195) < 1e-10)
      tmp1194 = pow(tmp1192, tmp1196);
    else
    {
      tmp1198 = modf(1.0/tmp1193, &tmp1197);
      if(tmp1198 > 0.5)
      {
        tmp1198 -= 1.0;
        tmp1197 += 1.0;
      }
      else if(tmp1198 < -0.5)
      {
        tmp1198 += 1.0;
        tmp1197 -= 1.0;
      }
      if(fabs(tmp1198) < 1e-10 && ((unsigned long)tmp1197 & 1))
      {
        tmp1194 = -pow(-tmp1192, tmp1195)*pow(tmp1192, tmp1196);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1192, tmp1193);
      }
    }
  }
  else
  {
    tmp1194 = pow(tmp1192, tmp1193);
  }
  if(isnan(tmp1194) || isinf(tmp1194))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1192, tmp1193);
  }tmp1199 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1194,"(r_init[381] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1199 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[381] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1199);
    }
  }
  (data->simulationInfo->realParameter[885] /* omega_c[381] PARAM */) = sqrt(tmp1199);
  TRACE_POP
}

/*
equation index: 13241
type: SIMPLE_ASSIGN
r_init[380] = r_min + 380.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13241};
  (data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (380.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13242
type: SIMPLE_ASSIGN
omega_c[380] = sqrt(G * Md / (r_init[380] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13242(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13242};
  modelica_real tmp1200;
  modelica_real tmp1201;
  modelica_real tmp1202;
  modelica_real tmp1203;
  modelica_real tmp1204;
  modelica_real tmp1205;
  modelica_real tmp1206;
  modelica_real tmp1207;
  modelica_real tmp1208;
  modelica_real tmp1209;
  tmp1200 = (data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */);
  tmp1201 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1202 = (tmp1200 * tmp1200) + (tmp1201 * tmp1201);
  tmp1203 = 1.5;
  if(tmp1202 < 0.0 && tmp1203 != 0.0)
  {
    tmp1205 = modf(tmp1203, &tmp1206);
    
    if(tmp1205 > 0.5)
    {
      tmp1205 -= 1.0;
      tmp1206 += 1.0;
    }
    else if(tmp1205 < -0.5)
    {
      tmp1205 += 1.0;
      tmp1206 -= 1.0;
    }
    
    if(fabs(tmp1205) < 1e-10)
      tmp1204 = pow(tmp1202, tmp1206);
    else
    {
      tmp1208 = modf(1.0/tmp1203, &tmp1207);
      if(tmp1208 > 0.5)
      {
        tmp1208 -= 1.0;
        tmp1207 += 1.0;
      }
      else if(tmp1208 < -0.5)
      {
        tmp1208 += 1.0;
        tmp1207 -= 1.0;
      }
      if(fabs(tmp1208) < 1e-10 && ((unsigned long)tmp1207 & 1))
      {
        tmp1204 = -pow(-tmp1202, tmp1205)*pow(tmp1202, tmp1206);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1202, tmp1203);
      }
    }
  }
  else
  {
    tmp1204 = pow(tmp1202, tmp1203);
  }
  if(isnan(tmp1204) || isinf(tmp1204))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1202, tmp1203);
  }tmp1209 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1204,"(r_init[380] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1209 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[380] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1209);
    }
  }
  (data->simulationInfo->realParameter[884] /* omega_c[380] PARAM */) = sqrt(tmp1209);
  TRACE_POP
}

/*
equation index: 13243
type: SIMPLE_ASSIGN
r_init[379] = r_min + 379.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13243};
  (data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (379.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13244
type: SIMPLE_ASSIGN
omega_c[379] = sqrt(G * Md / (r_init[379] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13244};
  modelica_real tmp1210;
  modelica_real tmp1211;
  modelica_real tmp1212;
  modelica_real tmp1213;
  modelica_real tmp1214;
  modelica_real tmp1215;
  modelica_real tmp1216;
  modelica_real tmp1217;
  modelica_real tmp1218;
  modelica_real tmp1219;
  tmp1210 = (data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */);
  tmp1211 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1212 = (tmp1210 * tmp1210) + (tmp1211 * tmp1211);
  tmp1213 = 1.5;
  if(tmp1212 < 0.0 && tmp1213 != 0.0)
  {
    tmp1215 = modf(tmp1213, &tmp1216);
    
    if(tmp1215 > 0.5)
    {
      tmp1215 -= 1.0;
      tmp1216 += 1.0;
    }
    else if(tmp1215 < -0.5)
    {
      tmp1215 += 1.0;
      tmp1216 -= 1.0;
    }
    
    if(fabs(tmp1215) < 1e-10)
      tmp1214 = pow(tmp1212, tmp1216);
    else
    {
      tmp1218 = modf(1.0/tmp1213, &tmp1217);
      if(tmp1218 > 0.5)
      {
        tmp1218 -= 1.0;
        tmp1217 += 1.0;
      }
      else if(tmp1218 < -0.5)
      {
        tmp1218 += 1.0;
        tmp1217 -= 1.0;
      }
      if(fabs(tmp1218) < 1e-10 && ((unsigned long)tmp1217 & 1))
      {
        tmp1214 = -pow(-tmp1212, tmp1215)*pow(tmp1212, tmp1216);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1212, tmp1213);
      }
    }
  }
  else
  {
    tmp1214 = pow(tmp1212, tmp1213);
  }
  if(isnan(tmp1214) || isinf(tmp1214))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1212, tmp1213);
  }tmp1219 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1214,"(r_init[379] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1219 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[379] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1219);
    }
  }
  (data->simulationInfo->realParameter[883] /* omega_c[379] PARAM */) = sqrt(tmp1219);
  TRACE_POP
}

/*
equation index: 13245
type: SIMPLE_ASSIGN
r_init[378] = r_min + 378.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13245};
  (data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (378.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13246
type: SIMPLE_ASSIGN
omega_c[378] = sqrt(G * Md / (r_init[378] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13246(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13246};
  modelica_real tmp1220;
  modelica_real tmp1221;
  modelica_real tmp1222;
  modelica_real tmp1223;
  modelica_real tmp1224;
  modelica_real tmp1225;
  modelica_real tmp1226;
  modelica_real tmp1227;
  modelica_real tmp1228;
  modelica_real tmp1229;
  tmp1220 = (data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */);
  tmp1221 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1222 = (tmp1220 * tmp1220) + (tmp1221 * tmp1221);
  tmp1223 = 1.5;
  if(tmp1222 < 0.0 && tmp1223 != 0.0)
  {
    tmp1225 = modf(tmp1223, &tmp1226);
    
    if(tmp1225 > 0.5)
    {
      tmp1225 -= 1.0;
      tmp1226 += 1.0;
    }
    else if(tmp1225 < -0.5)
    {
      tmp1225 += 1.0;
      tmp1226 -= 1.0;
    }
    
    if(fabs(tmp1225) < 1e-10)
      tmp1224 = pow(tmp1222, tmp1226);
    else
    {
      tmp1228 = modf(1.0/tmp1223, &tmp1227);
      if(tmp1228 > 0.5)
      {
        tmp1228 -= 1.0;
        tmp1227 += 1.0;
      }
      else if(tmp1228 < -0.5)
      {
        tmp1228 += 1.0;
        tmp1227 -= 1.0;
      }
      if(fabs(tmp1228) < 1e-10 && ((unsigned long)tmp1227 & 1))
      {
        tmp1224 = -pow(-tmp1222, tmp1225)*pow(tmp1222, tmp1226);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1222, tmp1223);
      }
    }
  }
  else
  {
    tmp1224 = pow(tmp1222, tmp1223);
  }
  if(isnan(tmp1224) || isinf(tmp1224))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1222, tmp1223);
  }tmp1229 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1224,"(r_init[378] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1229 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[378] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1229);
    }
  }
  (data->simulationInfo->realParameter[882] /* omega_c[378] PARAM */) = sqrt(tmp1229);
  TRACE_POP
}

/*
equation index: 13247
type: SIMPLE_ASSIGN
r_init[377] = r_min + 377.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13247};
  (data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (377.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13248
type: SIMPLE_ASSIGN
omega_c[377] = sqrt(G * Md / (r_init[377] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13248(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13248};
  modelica_real tmp1230;
  modelica_real tmp1231;
  modelica_real tmp1232;
  modelica_real tmp1233;
  modelica_real tmp1234;
  modelica_real tmp1235;
  modelica_real tmp1236;
  modelica_real tmp1237;
  modelica_real tmp1238;
  modelica_real tmp1239;
  tmp1230 = (data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */);
  tmp1231 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1232 = (tmp1230 * tmp1230) + (tmp1231 * tmp1231);
  tmp1233 = 1.5;
  if(tmp1232 < 0.0 && tmp1233 != 0.0)
  {
    tmp1235 = modf(tmp1233, &tmp1236);
    
    if(tmp1235 > 0.5)
    {
      tmp1235 -= 1.0;
      tmp1236 += 1.0;
    }
    else if(tmp1235 < -0.5)
    {
      tmp1235 += 1.0;
      tmp1236 -= 1.0;
    }
    
    if(fabs(tmp1235) < 1e-10)
      tmp1234 = pow(tmp1232, tmp1236);
    else
    {
      tmp1238 = modf(1.0/tmp1233, &tmp1237);
      if(tmp1238 > 0.5)
      {
        tmp1238 -= 1.0;
        tmp1237 += 1.0;
      }
      else if(tmp1238 < -0.5)
      {
        tmp1238 += 1.0;
        tmp1237 -= 1.0;
      }
      if(fabs(tmp1238) < 1e-10 && ((unsigned long)tmp1237 & 1))
      {
        tmp1234 = -pow(-tmp1232, tmp1235)*pow(tmp1232, tmp1236);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1232, tmp1233);
      }
    }
  }
  else
  {
    tmp1234 = pow(tmp1232, tmp1233);
  }
  if(isnan(tmp1234) || isinf(tmp1234))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1232, tmp1233);
  }tmp1239 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1234,"(r_init[377] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1239 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[377] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1239);
    }
  }
  (data->simulationInfo->realParameter[881] /* omega_c[377] PARAM */) = sqrt(tmp1239);
  TRACE_POP
}

/*
equation index: 13249
type: SIMPLE_ASSIGN
r_init[376] = r_min + 376.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13249};
  (data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (376.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13250
type: SIMPLE_ASSIGN
omega_c[376] = sqrt(G * Md / (r_init[376] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13250(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13250};
  modelica_real tmp1240;
  modelica_real tmp1241;
  modelica_real tmp1242;
  modelica_real tmp1243;
  modelica_real tmp1244;
  modelica_real tmp1245;
  modelica_real tmp1246;
  modelica_real tmp1247;
  modelica_real tmp1248;
  modelica_real tmp1249;
  tmp1240 = (data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */);
  tmp1241 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1242 = (tmp1240 * tmp1240) + (tmp1241 * tmp1241);
  tmp1243 = 1.5;
  if(tmp1242 < 0.0 && tmp1243 != 0.0)
  {
    tmp1245 = modf(tmp1243, &tmp1246);
    
    if(tmp1245 > 0.5)
    {
      tmp1245 -= 1.0;
      tmp1246 += 1.0;
    }
    else if(tmp1245 < -0.5)
    {
      tmp1245 += 1.0;
      tmp1246 -= 1.0;
    }
    
    if(fabs(tmp1245) < 1e-10)
      tmp1244 = pow(tmp1242, tmp1246);
    else
    {
      tmp1248 = modf(1.0/tmp1243, &tmp1247);
      if(tmp1248 > 0.5)
      {
        tmp1248 -= 1.0;
        tmp1247 += 1.0;
      }
      else if(tmp1248 < -0.5)
      {
        tmp1248 += 1.0;
        tmp1247 -= 1.0;
      }
      if(fabs(tmp1248) < 1e-10 && ((unsigned long)tmp1247 & 1))
      {
        tmp1244 = -pow(-tmp1242, tmp1245)*pow(tmp1242, tmp1246);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1242, tmp1243);
      }
    }
  }
  else
  {
    tmp1244 = pow(tmp1242, tmp1243);
  }
  if(isnan(tmp1244) || isinf(tmp1244))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1242, tmp1243);
  }tmp1249 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1244,"(r_init[376] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1249 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[376] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1249);
    }
  }
  (data->simulationInfo->realParameter[880] /* omega_c[376] PARAM */) = sqrt(tmp1249);
  TRACE_POP
}

/*
equation index: 13251
type: SIMPLE_ASSIGN
r_init[375] = r_min + 375.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13251};
  (data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (375.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13252
type: SIMPLE_ASSIGN
omega_c[375] = sqrt(G * Md / (r_init[375] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13252};
  modelica_real tmp1250;
  modelica_real tmp1251;
  modelica_real tmp1252;
  modelica_real tmp1253;
  modelica_real tmp1254;
  modelica_real tmp1255;
  modelica_real tmp1256;
  modelica_real tmp1257;
  modelica_real tmp1258;
  modelica_real tmp1259;
  tmp1250 = (data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */);
  tmp1251 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1252 = (tmp1250 * tmp1250) + (tmp1251 * tmp1251);
  tmp1253 = 1.5;
  if(tmp1252 < 0.0 && tmp1253 != 0.0)
  {
    tmp1255 = modf(tmp1253, &tmp1256);
    
    if(tmp1255 > 0.5)
    {
      tmp1255 -= 1.0;
      tmp1256 += 1.0;
    }
    else if(tmp1255 < -0.5)
    {
      tmp1255 += 1.0;
      tmp1256 -= 1.0;
    }
    
    if(fabs(tmp1255) < 1e-10)
      tmp1254 = pow(tmp1252, tmp1256);
    else
    {
      tmp1258 = modf(1.0/tmp1253, &tmp1257);
      if(tmp1258 > 0.5)
      {
        tmp1258 -= 1.0;
        tmp1257 += 1.0;
      }
      else if(tmp1258 < -0.5)
      {
        tmp1258 += 1.0;
        tmp1257 -= 1.0;
      }
      if(fabs(tmp1258) < 1e-10 && ((unsigned long)tmp1257 & 1))
      {
        tmp1254 = -pow(-tmp1252, tmp1255)*pow(tmp1252, tmp1256);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1252, tmp1253);
      }
    }
  }
  else
  {
    tmp1254 = pow(tmp1252, tmp1253);
  }
  if(isnan(tmp1254) || isinf(tmp1254))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1252, tmp1253);
  }tmp1259 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1254,"(r_init[375] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1259 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[375] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1259);
    }
  }
  (data->simulationInfo->realParameter[879] /* omega_c[375] PARAM */) = sqrt(tmp1259);
  TRACE_POP
}

/*
equation index: 13253
type: SIMPLE_ASSIGN
r_init[374] = r_min + 374.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13253};
  (data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (374.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13254
type: SIMPLE_ASSIGN
omega_c[374] = sqrt(G * Md / (r_init[374] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13254(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13254};
  modelica_real tmp1260;
  modelica_real tmp1261;
  modelica_real tmp1262;
  modelica_real tmp1263;
  modelica_real tmp1264;
  modelica_real tmp1265;
  modelica_real tmp1266;
  modelica_real tmp1267;
  modelica_real tmp1268;
  modelica_real tmp1269;
  tmp1260 = (data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */);
  tmp1261 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1262 = (tmp1260 * tmp1260) + (tmp1261 * tmp1261);
  tmp1263 = 1.5;
  if(tmp1262 < 0.0 && tmp1263 != 0.0)
  {
    tmp1265 = modf(tmp1263, &tmp1266);
    
    if(tmp1265 > 0.5)
    {
      tmp1265 -= 1.0;
      tmp1266 += 1.0;
    }
    else if(tmp1265 < -0.5)
    {
      tmp1265 += 1.0;
      tmp1266 -= 1.0;
    }
    
    if(fabs(tmp1265) < 1e-10)
      tmp1264 = pow(tmp1262, tmp1266);
    else
    {
      tmp1268 = modf(1.0/tmp1263, &tmp1267);
      if(tmp1268 > 0.5)
      {
        tmp1268 -= 1.0;
        tmp1267 += 1.0;
      }
      else if(tmp1268 < -0.5)
      {
        tmp1268 += 1.0;
        tmp1267 -= 1.0;
      }
      if(fabs(tmp1268) < 1e-10 && ((unsigned long)tmp1267 & 1))
      {
        tmp1264 = -pow(-tmp1262, tmp1265)*pow(tmp1262, tmp1266);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1262, tmp1263);
      }
    }
  }
  else
  {
    tmp1264 = pow(tmp1262, tmp1263);
  }
  if(isnan(tmp1264) || isinf(tmp1264))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1262, tmp1263);
  }tmp1269 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1264,"(r_init[374] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1269 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[374] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1269);
    }
  }
  (data->simulationInfo->realParameter[878] /* omega_c[374] PARAM */) = sqrt(tmp1269);
  TRACE_POP
}

/*
equation index: 13255
type: SIMPLE_ASSIGN
r_init[373] = r_min + 373.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13255};
  (data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (373.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13256
type: SIMPLE_ASSIGN
omega_c[373] = sqrt(G * Md / (r_init[373] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13256(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13256};
  modelica_real tmp1270;
  modelica_real tmp1271;
  modelica_real tmp1272;
  modelica_real tmp1273;
  modelica_real tmp1274;
  modelica_real tmp1275;
  modelica_real tmp1276;
  modelica_real tmp1277;
  modelica_real tmp1278;
  modelica_real tmp1279;
  tmp1270 = (data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */);
  tmp1271 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1272 = (tmp1270 * tmp1270) + (tmp1271 * tmp1271);
  tmp1273 = 1.5;
  if(tmp1272 < 0.0 && tmp1273 != 0.0)
  {
    tmp1275 = modf(tmp1273, &tmp1276);
    
    if(tmp1275 > 0.5)
    {
      tmp1275 -= 1.0;
      tmp1276 += 1.0;
    }
    else if(tmp1275 < -0.5)
    {
      tmp1275 += 1.0;
      tmp1276 -= 1.0;
    }
    
    if(fabs(tmp1275) < 1e-10)
      tmp1274 = pow(tmp1272, tmp1276);
    else
    {
      tmp1278 = modf(1.0/tmp1273, &tmp1277);
      if(tmp1278 > 0.5)
      {
        tmp1278 -= 1.0;
        tmp1277 += 1.0;
      }
      else if(tmp1278 < -0.5)
      {
        tmp1278 += 1.0;
        tmp1277 -= 1.0;
      }
      if(fabs(tmp1278) < 1e-10 && ((unsigned long)tmp1277 & 1))
      {
        tmp1274 = -pow(-tmp1272, tmp1275)*pow(tmp1272, tmp1276);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1272, tmp1273);
      }
    }
  }
  else
  {
    tmp1274 = pow(tmp1272, tmp1273);
  }
  if(isnan(tmp1274) || isinf(tmp1274))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1272, tmp1273);
  }tmp1279 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1274,"(r_init[373] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1279 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[373] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1279);
    }
  }
  (data->simulationInfo->realParameter[877] /* omega_c[373] PARAM */) = sqrt(tmp1279);
  TRACE_POP
}

/*
equation index: 13257
type: SIMPLE_ASSIGN
r_init[372] = r_min + 372.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13257};
  (data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (372.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13258
type: SIMPLE_ASSIGN
omega_c[372] = sqrt(G * Md / (r_init[372] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13258(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13258};
  modelica_real tmp1280;
  modelica_real tmp1281;
  modelica_real tmp1282;
  modelica_real tmp1283;
  modelica_real tmp1284;
  modelica_real tmp1285;
  modelica_real tmp1286;
  modelica_real tmp1287;
  modelica_real tmp1288;
  modelica_real tmp1289;
  tmp1280 = (data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */);
  tmp1281 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1282 = (tmp1280 * tmp1280) + (tmp1281 * tmp1281);
  tmp1283 = 1.5;
  if(tmp1282 < 0.0 && tmp1283 != 0.0)
  {
    tmp1285 = modf(tmp1283, &tmp1286);
    
    if(tmp1285 > 0.5)
    {
      tmp1285 -= 1.0;
      tmp1286 += 1.0;
    }
    else if(tmp1285 < -0.5)
    {
      tmp1285 += 1.0;
      tmp1286 -= 1.0;
    }
    
    if(fabs(tmp1285) < 1e-10)
      tmp1284 = pow(tmp1282, tmp1286);
    else
    {
      tmp1288 = modf(1.0/tmp1283, &tmp1287);
      if(tmp1288 > 0.5)
      {
        tmp1288 -= 1.0;
        tmp1287 += 1.0;
      }
      else if(tmp1288 < -0.5)
      {
        tmp1288 += 1.0;
        tmp1287 -= 1.0;
      }
      if(fabs(tmp1288) < 1e-10 && ((unsigned long)tmp1287 & 1))
      {
        tmp1284 = -pow(-tmp1282, tmp1285)*pow(tmp1282, tmp1286);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1282, tmp1283);
      }
    }
  }
  else
  {
    tmp1284 = pow(tmp1282, tmp1283);
  }
  if(isnan(tmp1284) || isinf(tmp1284))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1282, tmp1283);
  }tmp1289 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1284,"(r_init[372] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1289 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[372] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1289);
    }
  }
  (data->simulationInfo->realParameter[876] /* omega_c[372] PARAM */) = sqrt(tmp1289);
  TRACE_POP
}

/*
equation index: 13259
type: SIMPLE_ASSIGN
r_init[371] = r_min + 371.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13259};
  (data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (371.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13260
type: SIMPLE_ASSIGN
omega_c[371] = sqrt(G * Md / (r_init[371] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13260};
  modelica_real tmp1290;
  modelica_real tmp1291;
  modelica_real tmp1292;
  modelica_real tmp1293;
  modelica_real tmp1294;
  modelica_real tmp1295;
  modelica_real tmp1296;
  modelica_real tmp1297;
  modelica_real tmp1298;
  modelica_real tmp1299;
  tmp1290 = (data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */);
  tmp1291 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1292 = (tmp1290 * tmp1290) + (tmp1291 * tmp1291);
  tmp1293 = 1.5;
  if(tmp1292 < 0.0 && tmp1293 != 0.0)
  {
    tmp1295 = modf(tmp1293, &tmp1296);
    
    if(tmp1295 > 0.5)
    {
      tmp1295 -= 1.0;
      tmp1296 += 1.0;
    }
    else if(tmp1295 < -0.5)
    {
      tmp1295 += 1.0;
      tmp1296 -= 1.0;
    }
    
    if(fabs(tmp1295) < 1e-10)
      tmp1294 = pow(tmp1292, tmp1296);
    else
    {
      tmp1298 = modf(1.0/tmp1293, &tmp1297);
      if(tmp1298 > 0.5)
      {
        tmp1298 -= 1.0;
        tmp1297 += 1.0;
      }
      else if(tmp1298 < -0.5)
      {
        tmp1298 += 1.0;
        tmp1297 -= 1.0;
      }
      if(fabs(tmp1298) < 1e-10 && ((unsigned long)tmp1297 & 1))
      {
        tmp1294 = -pow(-tmp1292, tmp1295)*pow(tmp1292, tmp1296);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1292, tmp1293);
      }
    }
  }
  else
  {
    tmp1294 = pow(tmp1292, tmp1293);
  }
  if(isnan(tmp1294) || isinf(tmp1294))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1292, tmp1293);
  }tmp1299 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1294,"(r_init[371] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1299 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[371] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1299);
    }
  }
  (data->simulationInfo->realParameter[875] /* omega_c[371] PARAM */) = sqrt(tmp1299);
  TRACE_POP
}

/*
equation index: 13261
type: SIMPLE_ASSIGN
r_init[370] = r_min + 370.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13261};
  (data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (370.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13262
type: SIMPLE_ASSIGN
omega_c[370] = sqrt(G * Md / (r_init[370] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13262(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13262};
  modelica_real tmp1300;
  modelica_real tmp1301;
  modelica_real tmp1302;
  modelica_real tmp1303;
  modelica_real tmp1304;
  modelica_real tmp1305;
  modelica_real tmp1306;
  modelica_real tmp1307;
  modelica_real tmp1308;
  modelica_real tmp1309;
  tmp1300 = (data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */);
  tmp1301 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1302 = (tmp1300 * tmp1300) + (tmp1301 * tmp1301);
  tmp1303 = 1.5;
  if(tmp1302 < 0.0 && tmp1303 != 0.0)
  {
    tmp1305 = modf(tmp1303, &tmp1306);
    
    if(tmp1305 > 0.5)
    {
      tmp1305 -= 1.0;
      tmp1306 += 1.0;
    }
    else if(tmp1305 < -0.5)
    {
      tmp1305 += 1.0;
      tmp1306 -= 1.0;
    }
    
    if(fabs(tmp1305) < 1e-10)
      tmp1304 = pow(tmp1302, tmp1306);
    else
    {
      tmp1308 = modf(1.0/tmp1303, &tmp1307);
      if(tmp1308 > 0.5)
      {
        tmp1308 -= 1.0;
        tmp1307 += 1.0;
      }
      else if(tmp1308 < -0.5)
      {
        tmp1308 += 1.0;
        tmp1307 -= 1.0;
      }
      if(fabs(tmp1308) < 1e-10 && ((unsigned long)tmp1307 & 1))
      {
        tmp1304 = -pow(-tmp1302, tmp1305)*pow(tmp1302, tmp1306);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1302, tmp1303);
      }
    }
  }
  else
  {
    tmp1304 = pow(tmp1302, tmp1303);
  }
  if(isnan(tmp1304) || isinf(tmp1304))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1302, tmp1303);
  }tmp1309 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1304,"(r_init[370] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1309 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[370] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1309);
    }
  }
  (data->simulationInfo->realParameter[874] /* omega_c[370] PARAM */) = sqrt(tmp1309);
  TRACE_POP
}

/*
equation index: 13263
type: SIMPLE_ASSIGN
r_init[369] = r_min + 369.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13263};
  (data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (369.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13264
type: SIMPLE_ASSIGN
omega_c[369] = sqrt(G * Md / (r_init[369] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13264(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13264};
  modelica_real tmp1310;
  modelica_real tmp1311;
  modelica_real tmp1312;
  modelica_real tmp1313;
  modelica_real tmp1314;
  modelica_real tmp1315;
  modelica_real tmp1316;
  modelica_real tmp1317;
  modelica_real tmp1318;
  modelica_real tmp1319;
  tmp1310 = (data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */);
  tmp1311 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1312 = (tmp1310 * tmp1310) + (tmp1311 * tmp1311);
  tmp1313 = 1.5;
  if(tmp1312 < 0.0 && tmp1313 != 0.0)
  {
    tmp1315 = modf(tmp1313, &tmp1316);
    
    if(tmp1315 > 0.5)
    {
      tmp1315 -= 1.0;
      tmp1316 += 1.0;
    }
    else if(tmp1315 < -0.5)
    {
      tmp1315 += 1.0;
      tmp1316 -= 1.0;
    }
    
    if(fabs(tmp1315) < 1e-10)
      tmp1314 = pow(tmp1312, tmp1316);
    else
    {
      tmp1318 = modf(1.0/tmp1313, &tmp1317);
      if(tmp1318 > 0.5)
      {
        tmp1318 -= 1.0;
        tmp1317 += 1.0;
      }
      else if(tmp1318 < -0.5)
      {
        tmp1318 += 1.0;
        tmp1317 -= 1.0;
      }
      if(fabs(tmp1318) < 1e-10 && ((unsigned long)tmp1317 & 1))
      {
        tmp1314 = -pow(-tmp1312, tmp1315)*pow(tmp1312, tmp1316);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1312, tmp1313);
      }
    }
  }
  else
  {
    tmp1314 = pow(tmp1312, tmp1313);
  }
  if(isnan(tmp1314) || isinf(tmp1314))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1312, tmp1313);
  }tmp1319 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1314,"(r_init[369] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1319 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[369] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1319);
    }
  }
  (data->simulationInfo->realParameter[873] /* omega_c[369] PARAM */) = sqrt(tmp1319);
  TRACE_POP
}

/*
equation index: 13265
type: SIMPLE_ASSIGN
r_init[368] = r_min + 368.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13265};
  (data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (368.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13266
type: SIMPLE_ASSIGN
omega_c[368] = sqrt(G * Md / (r_init[368] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13266(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13266};
  modelica_real tmp1320;
  modelica_real tmp1321;
  modelica_real tmp1322;
  modelica_real tmp1323;
  modelica_real tmp1324;
  modelica_real tmp1325;
  modelica_real tmp1326;
  modelica_real tmp1327;
  modelica_real tmp1328;
  modelica_real tmp1329;
  tmp1320 = (data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */);
  tmp1321 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1322 = (tmp1320 * tmp1320) + (tmp1321 * tmp1321);
  tmp1323 = 1.5;
  if(tmp1322 < 0.0 && tmp1323 != 0.0)
  {
    tmp1325 = modf(tmp1323, &tmp1326);
    
    if(tmp1325 > 0.5)
    {
      tmp1325 -= 1.0;
      tmp1326 += 1.0;
    }
    else if(tmp1325 < -0.5)
    {
      tmp1325 += 1.0;
      tmp1326 -= 1.0;
    }
    
    if(fabs(tmp1325) < 1e-10)
      tmp1324 = pow(tmp1322, tmp1326);
    else
    {
      tmp1328 = modf(1.0/tmp1323, &tmp1327);
      if(tmp1328 > 0.5)
      {
        tmp1328 -= 1.0;
        tmp1327 += 1.0;
      }
      else if(tmp1328 < -0.5)
      {
        tmp1328 += 1.0;
        tmp1327 -= 1.0;
      }
      if(fabs(tmp1328) < 1e-10 && ((unsigned long)tmp1327 & 1))
      {
        tmp1324 = -pow(-tmp1322, tmp1325)*pow(tmp1322, tmp1326);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1322, tmp1323);
      }
    }
  }
  else
  {
    tmp1324 = pow(tmp1322, tmp1323);
  }
  if(isnan(tmp1324) || isinf(tmp1324))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1322, tmp1323);
  }tmp1329 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1324,"(r_init[368] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1329 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[368] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1329);
    }
  }
  (data->simulationInfo->realParameter[872] /* omega_c[368] PARAM */) = sqrt(tmp1329);
  TRACE_POP
}

/*
equation index: 13267
type: SIMPLE_ASSIGN
r_init[367] = r_min + 367.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13267};
  (data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (367.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13268
type: SIMPLE_ASSIGN
omega_c[367] = sqrt(G * Md / (r_init[367] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13268};
  modelica_real tmp1330;
  modelica_real tmp1331;
  modelica_real tmp1332;
  modelica_real tmp1333;
  modelica_real tmp1334;
  modelica_real tmp1335;
  modelica_real tmp1336;
  modelica_real tmp1337;
  modelica_real tmp1338;
  modelica_real tmp1339;
  tmp1330 = (data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */);
  tmp1331 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1332 = (tmp1330 * tmp1330) + (tmp1331 * tmp1331);
  tmp1333 = 1.5;
  if(tmp1332 < 0.0 && tmp1333 != 0.0)
  {
    tmp1335 = modf(tmp1333, &tmp1336);
    
    if(tmp1335 > 0.5)
    {
      tmp1335 -= 1.0;
      tmp1336 += 1.0;
    }
    else if(tmp1335 < -0.5)
    {
      tmp1335 += 1.0;
      tmp1336 -= 1.0;
    }
    
    if(fabs(tmp1335) < 1e-10)
      tmp1334 = pow(tmp1332, tmp1336);
    else
    {
      tmp1338 = modf(1.0/tmp1333, &tmp1337);
      if(tmp1338 > 0.5)
      {
        tmp1338 -= 1.0;
        tmp1337 += 1.0;
      }
      else if(tmp1338 < -0.5)
      {
        tmp1338 += 1.0;
        tmp1337 -= 1.0;
      }
      if(fabs(tmp1338) < 1e-10 && ((unsigned long)tmp1337 & 1))
      {
        tmp1334 = -pow(-tmp1332, tmp1335)*pow(tmp1332, tmp1336);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1332, tmp1333);
      }
    }
  }
  else
  {
    tmp1334 = pow(tmp1332, tmp1333);
  }
  if(isnan(tmp1334) || isinf(tmp1334))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1332, tmp1333);
  }tmp1339 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1334,"(r_init[367] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1339 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[367] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1339);
    }
  }
  (data->simulationInfo->realParameter[871] /* omega_c[367] PARAM */) = sqrt(tmp1339);
  TRACE_POP
}

/*
equation index: 13269
type: SIMPLE_ASSIGN
r_init[366] = r_min + 366.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13269};
  (data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (366.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13270
type: SIMPLE_ASSIGN
omega_c[366] = sqrt(G * Md / (r_init[366] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13270(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13270};
  modelica_real tmp1340;
  modelica_real tmp1341;
  modelica_real tmp1342;
  modelica_real tmp1343;
  modelica_real tmp1344;
  modelica_real tmp1345;
  modelica_real tmp1346;
  modelica_real tmp1347;
  modelica_real tmp1348;
  modelica_real tmp1349;
  tmp1340 = (data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */);
  tmp1341 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1342 = (tmp1340 * tmp1340) + (tmp1341 * tmp1341);
  tmp1343 = 1.5;
  if(tmp1342 < 0.0 && tmp1343 != 0.0)
  {
    tmp1345 = modf(tmp1343, &tmp1346);
    
    if(tmp1345 > 0.5)
    {
      tmp1345 -= 1.0;
      tmp1346 += 1.0;
    }
    else if(tmp1345 < -0.5)
    {
      tmp1345 += 1.0;
      tmp1346 -= 1.0;
    }
    
    if(fabs(tmp1345) < 1e-10)
      tmp1344 = pow(tmp1342, tmp1346);
    else
    {
      tmp1348 = modf(1.0/tmp1343, &tmp1347);
      if(tmp1348 > 0.5)
      {
        tmp1348 -= 1.0;
        tmp1347 += 1.0;
      }
      else if(tmp1348 < -0.5)
      {
        tmp1348 += 1.0;
        tmp1347 -= 1.0;
      }
      if(fabs(tmp1348) < 1e-10 && ((unsigned long)tmp1347 & 1))
      {
        tmp1344 = -pow(-tmp1342, tmp1345)*pow(tmp1342, tmp1346);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1342, tmp1343);
      }
    }
  }
  else
  {
    tmp1344 = pow(tmp1342, tmp1343);
  }
  if(isnan(tmp1344) || isinf(tmp1344))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1342, tmp1343);
  }tmp1349 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1344,"(r_init[366] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1349 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[366] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1349);
    }
  }
  (data->simulationInfo->realParameter[870] /* omega_c[366] PARAM */) = sqrt(tmp1349);
  TRACE_POP
}

/*
equation index: 13271
type: SIMPLE_ASSIGN
r_init[365] = r_min + 365.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13271};
  (data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (365.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13272
type: SIMPLE_ASSIGN
omega_c[365] = sqrt(G * Md / (r_init[365] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13272(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13272};
  modelica_real tmp1350;
  modelica_real tmp1351;
  modelica_real tmp1352;
  modelica_real tmp1353;
  modelica_real tmp1354;
  modelica_real tmp1355;
  modelica_real tmp1356;
  modelica_real tmp1357;
  modelica_real tmp1358;
  modelica_real tmp1359;
  tmp1350 = (data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */);
  tmp1351 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1352 = (tmp1350 * tmp1350) + (tmp1351 * tmp1351);
  tmp1353 = 1.5;
  if(tmp1352 < 0.0 && tmp1353 != 0.0)
  {
    tmp1355 = modf(tmp1353, &tmp1356);
    
    if(tmp1355 > 0.5)
    {
      tmp1355 -= 1.0;
      tmp1356 += 1.0;
    }
    else if(tmp1355 < -0.5)
    {
      tmp1355 += 1.0;
      tmp1356 -= 1.0;
    }
    
    if(fabs(tmp1355) < 1e-10)
      tmp1354 = pow(tmp1352, tmp1356);
    else
    {
      tmp1358 = modf(1.0/tmp1353, &tmp1357);
      if(tmp1358 > 0.5)
      {
        tmp1358 -= 1.0;
        tmp1357 += 1.0;
      }
      else if(tmp1358 < -0.5)
      {
        tmp1358 += 1.0;
        tmp1357 -= 1.0;
      }
      if(fabs(tmp1358) < 1e-10 && ((unsigned long)tmp1357 & 1))
      {
        tmp1354 = -pow(-tmp1352, tmp1355)*pow(tmp1352, tmp1356);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1352, tmp1353);
      }
    }
  }
  else
  {
    tmp1354 = pow(tmp1352, tmp1353);
  }
  if(isnan(tmp1354) || isinf(tmp1354))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1352, tmp1353);
  }tmp1359 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1354,"(r_init[365] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1359 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[365] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1359);
    }
  }
  (data->simulationInfo->realParameter[869] /* omega_c[365] PARAM */) = sqrt(tmp1359);
  TRACE_POP
}

/*
equation index: 13273
type: SIMPLE_ASSIGN
r_init[364] = r_min + 364.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13273};
  (data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (364.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13274
type: SIMPLE_ASSIGN
omega_c[364] = sqrt(G * Md / (r_init[364] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13274(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13274};
  modelica_real tmp1360;
  modelica_real tmp1361;
  modelica_real tmp1362;
  modelica_real tmp1363;
  modelica_real tmp1364;
  modelica_real tmp1365;
  modelica_real tmp1366;
  modelica_real tmp1367;
  modelica_real tmp1368;
  modelica_real tmp1369;
  tmp1360 = (data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */);
  tmp1361 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1362 = (tmp1360 * tmp1360) + (tmp1361 * tmp1361);
  tmp1363 = 1.5;
  if(tmp1362 < 0.0 && tmp1363 != 0.0)
  {
    tmp1365 = modf(tmp1363, &tmp1366);
    
    if(tmp1365 > 0.5)
    {
      tmp1365 -= 1.0;
      tmp1366 += 1.0;
    }
    else if(tmp1365 < -0.5)
    {
      tmp1365 += 1.0;
      tmp1366 -= 1.0;
    }
    
    if(fabs(tmp1365) < 1e-10)
      tmp1364 = pow(tmp1362, tmp1366);
    else
    {
      tmp1368 = modf(1.0/tmp1363, &tmp1367);
      if(tmp1368 > 0.5)
      {
        tmp1368 -= 1.0;
        tmp1367 += 1.0;
      }
      else if(tmp1368 < -0.5)
      {
        tmp1368 += 1.0;
        tmp1367 -= 1.0;
      }
      if(fabs(tmp1368) < 1e-10 && ((unsigned long)tmp1367 & 1))
      {
        tmp1364 = -pow(-tmp1362, tmp1365)*pow(tmp1362, tmp1366);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1362, tmp1363);
      }
    }
  }
  else
  {
    tmp1364 = pow(tmp1362, tmp1363);
  }
  if(isnan(tmp1364) || isinf(tmp1364))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1362, tmp1363);
  }tmp1369 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1364,"(r_init[364] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1369 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[364] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1369);
    }
  }
  (data->simulationInfo->realParameter[868] /* omega_c[364] PARAM */) = sqrt(tmp1369);
  TRACE_POP
}

/*
equation index: 13275
type: SIMPLE_ASSIGN
r_init[363] = r_min + 363.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13275};
  (data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (363.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13276
type: SIMPLE_ASSIGN
omega_c[363] = sqrt(G * Md / (r_init[363] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13276};
  modelica_real tmp1370;
  modelica_real tmp1371;
  modelica_real tmp1372;
  modelica_real tmp1373;
  modelica_real tmp1374;
  modelica_real tmp1375;
  modelica_real tmp1376;
  modelica_real tmp1377;
  modelica_real tmp1378;
  modelica_real tmp1379;
  tmp1370 = (data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */);
  tmp1371 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1372 = (tmp1370 * tmp1370) + (tmp1371 * tmp1371);
  tmp1373 = 1.5;
  if(tmp1372 < 0.0 && tmp1373 != 0.0)
  {
    tmp1375 = modf(tmp1373, &tmp1376);
    
    if(tmp1375 > 0.5)
    {
      tmp1375 -= 1.0;
      tmp1376 += 1.0;
    }
    else if(tmp1375 < -0.5)
    {
      tmp1375 += 1.0;
      tmp1376 -= 1.0;
    }
    
    if(fabs(tmp1375) < 1e-10)
      tmp1374 = pow(tmp1372, tmp1376);
    else
    {
      tmp1378 = modf(1.0/tmp1373, &tmp1377);
      if(tmp1378 > 0.5)
      {
        tmp1378 -= 1.0;
        tmp1377 += 1.0;
      }
      else if(tmp1378 < -0.5)
      {
        tmp1378 += 1.0;
        tmp1377 -= 1.0;
      }
      if(fabs(tmp1378) < 1e-10 && ((unsigned long)tmp1377 & 1))
      {
        tmp1374 = -pow(-tmp1372, tmp1375)*pow(tmp1372, tmp1376);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1372, tmp1373);
      }
    }
  }
  else
  {
    tmp1374 = pow(tmp1372, tmp1373);
  }
  if(isnan(tmp1374) || isinf(tmp1374))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1372, tmp1373);
  }tmp1379 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1374,"(r_init[363] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1379 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[363] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1379);
    }
  }
  (data->simulationInfo->realParameter[867] /* omega_c[363] PARAM */) = sqrt(tmp1379);
  TRACE_POP
}

/*
equation index: 13277
type: SIMPLE_ASSIGN
r_init[362] = r_min + 362.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13277};
  (data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (362.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13278
type: SIMPLE_ASSIGN
omega_c[362] = sqrt(G * Md / (r_init[362] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13278(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13278};
  modelica_real tmp1380;
  modelica_real tmp1381;
  modelica_real tmp1382;
  modelica_real tmp1383;
  modelica_real tmp1384;
  modelica_real tmp1385;
  modelica_real tmp1386;
  modelica_real tmp1387;
  modelica_real tmp1388;
  modelica_real tmp1389;
  tmp1380 = (data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */);
  tmp1381 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1382 = (tmp1380 * tmp1380) + (tmp1381 * tmp1381);
  tmp1383 = 1.5;
  if(tmp1382 < 0.0 && tmp1383 != 0.0)
  {
    tmp1385 = modf(tmp1383, &tmp1386);
    
    if(tmp1385 > 0.5)
    {
      tmp1385 -= 1.0;
      tmp1386 += 1.0;
    }
    else if(tmp1385 < -0.5)
    {
      tmp1385 += 1.0;
      tmp1386 -= 1.0;
    }
    
    if(fabs(tmp1385) < 1e-10)
      tmp1384 = pow(tmp1382, tmp1386);
    else
    {
      tmp1388 = modf(1.0/tmp1383, &tmp1387);
      if(tmp1388 > 0.5)
      {
        tmp1388 -= 1.0;
        tmp1387 += 1.0;
      }
      else if(tmp1388 < -0.5)
      {
        tmp1388 += 1.0;
        tmp1387 -= 1.0;
      }
      if(fabs(tmp1388) < 1e-10 && ((unsigned long)tmp1387 & 1))
      {
        tmp1384 = -pow(-tmp1382, tmp1385)*pow(tmp1382, tmp1386);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1382, tmp1383);
      }
    }
  }
  else
  {
    tmp1384 = pow(tmp1382, tmp1383);
  }
  if(isnan(tmp1384) || isinf(tmp1384))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1382, tmp1383);
  }tmp1389 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1384,"(r_init[362] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1389 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[362] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1389);
    }
  }
  (data->simulationInfo->realParameter[866] /* omega_c[362] PARAM */) = sqrt(tmp1389);
  TRACE_POP
}

/*
equation index: 13279
type: SIMPLE_ASSIGN
r_init[361] = r_min + 361.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13279};
  (data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (361.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13280
type: SIMPLE_ASSIGN
omega_c[361] = sqrt(G * Md / (r_init[361] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13280(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13280};
  modelica_real tmp1390;
  modelica_real tmp1391;
  modelica_real tmp1392;
  modelica_real tmp1393;
  modelica_real tmp1394;
  modelica_real tmp1395;
  modelica_real tmp1396;
  modelica_real tmp1397;
  modelica_real tmp1398;
  modelica_real tmp1399;
  tmp1390 = (data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */);
  tmp1391 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1392 = (tmp1390 * tmp1390) + (tmp1391 * tmp1391);
  tmp1393 = 1.5;
  if(tmp1392 < 0.0 && tmp1393 != 0.0)
  {
    tmp1395 = modf(tmp1393, &tmp1396);
    
    if(tmp1395 > 0.5)
    {
      tmp1395 -= 1.0;
      tmp1396 += 1.0;
    }
    else if(tmp1395 < -0.5)
    {
      tmp1395 += 1.0;
      tmp1396 -= 1.0;
    }
    
    if(fabs(tmp1395) < 1e-10)
      tmp1394 = pow(tmp1392, tmp1396);
    else
    {
      tmp1398 = modf(1.0/tmp1393, &tmp1397);
      if(tmp1398 > 0.5)
      {
        tmp1398 -= 1.0;
        tmp1397 += 1.0;
      }
      else if(tmp1398 < -0.5)
      {
        tmp1398 += 1.0;
        tmp1397 -= 1.0;
      }
      if(fabs(tmp1398) < 1e-10 && ((unsigned long)tmp1397 & 1))
      {
        tmp1394 = -pow(-tmp1392, tmp1395)*pow(tmp1392, tmp1396);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1392, tmp1393);
      }
    }
  }
  else
  {
    tmp1394 = pow(tmp1392, tmp1393);
  }
  if(isnan(tmp1394) || isinf(tmp1394))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1392, tmp1393);
  }tmp1399 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1394,"(r_init[361] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1399 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[361] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1399);
    }
  }
  (data->simulationInfo->realParameter[865] /* omega_c[361] PARAM */) = sqrt(tmp1399);
  TRACE_POP
}

/*
equation index: 13281
type: SIMPLE_ASSIGN
r_init[360] = r_min + 360.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13281};
  (data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (360.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13282
type: SIMPLE_ASSIGN
omega_c[360] = sqrt(G * Md / (r_init[360] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13282(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13282};
  modelica_real tmp1400;
  modelica_real tmp1401;
  modelica_real tmp1402;
  modelica_real tmp1403;
  modelica_real tmp1404;
  modelica_real tmp1405;
  modelica_real tmp1406;
  modelica_real tmp1407;
  modelica_real tmp1408;
  modelica_real tmp1409;
  tmp1400 = (data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */);
  tmp1401 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1402 = (tmp1400 * tmp1400) + (tmp1401 * tmp1401);
  tmp1403 = 1.5;
  if(tmp1402 < 0.0 && tmp1403 != 0.0)
  {
    tmp1405 = modf(tmp1403, &tmp1406);
    
    if(tmp1405 > 0.5)
    {
      tmp1405 -= 1.0;
      tmp1406 += 1.0;
    }
    else if(tmp1405 < -0.5)
    {
      tmp1405 += 1.0;
      tmp1406 -= 1.0;
    }
    
    if(fabs(tmp1405) < 1e-10)
      tmp1404 = pow(tmp1402, tmp1406);
    else
    {
      tmp1408 = modf(1.0/tmp1403, &tmp1407);
      if(tmp1408 > 0.5)
      {
        tmp1408 -= 1.0;
        tmp1407 += 1.0;
      }
      else if(tmp1408 < -0.5)
      {
        tmp1408 += 1.0;
        tmp1407 -= 1.0;
      }
      if(fabs(tmp1408) < 1e-10 && ((unsigned long)tmp1407 & 1))
      {
        tmp1404 = -pow(-tmp1402, tmp1405)*pow(tmp1402, tmp1406);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1402, tmp1403);
      }
    }
  }
  else
  {
    tmp1404 = pow(tmp1402, tmp1403);
  }
  if(isnan(tmp1404) || isinf(tmp1404))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1402, tmp1403);
  }tmp1409 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1404,"(r_init[360] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1409 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[360] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1409);
    }
  }
  (data->simulationInfo->realParameter[864] /* omega_c[360] PARAM */) = sqrt(tmp1409);
  TRACE_POP
}

/*
equation index: 13283
type: SIMPLE_ASSIGN
r_init[359] = r_min + 359.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13283};
  (data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (359.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13284
type: SIMPLE_ASSIGN
omega_c[359] = sqrt(G * Md / (r_init[359] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13284};
  modelica_real tmp1410;
  modelica_real tmp1411;
  modelica_real tmp1412;
  modelica_real tmp1413;
  modelica_real tmp1414;
  modelica_real tmp1415;
  modelica_real tmp1416;
  modelica_real tmp1417;
  modelica_real tmp1418;
  modelica_real tmp1419;
  tmp1410 = (data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */);
  tmp1411 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1412 = (tmp1410 * tmp1410) + (tmp1411 * tmp1411);
  tmp1413 = 1.5;
  if(tmp1412 < 0.0 && tmp1413 != 0.0)
  {
    tmp1415 = modf(tmp1413, &tmp1416);
    
    if(tmp1415 > 0.5)
    {
      tmp1415 -= 1.0;
      tmp1416 += 1.0;
    }
    else if(tmp1415 < -0.5)
    {
      tmp1415 += 1.0;
      tmp1416 -= 1.0;
    }
    
    if(fabs(tmp1415) < 1e-10)
      tmp1414 = pow(tmp1412, tmp1416);
    else
    {
      tmp1418 = modf(1.0/tmp1413, &tmp1417);
      if(tmp1418 > 0.5)
      {
        tmp1418 -= 1.0;
        tmp1417 += 1.0;
      }
      else if(tmp1418 < -0.5)
      {
        tmp1418 += 1.0;
        tmp1417 -= 1.0;
      }
      if(fabs(tmp1418) < 1e-10 && ((unsigned long)tmp1417 & 1))
      {
        tmp1414 = -pow(-tmp1412, tmp1415)*pow(tmp1412, tmp1416);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1412, tmp1413);
      }
    }
  }
  else
  {
    tmp1414 = pow(tmp1412, tmp1413);
  }
  if(isnan(tmp1414) || isinf(tmp1414))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1412, tmp1413);
  }tmp1419 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1414,"(r_init[359] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1419 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[359] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1419);
    }
  }
  (data->simulationInfo->realParameter[863] /* omega_c[359] PARAM */) = sqrt(tmp1419);
  TRACE_POP
}

/*
equation index: 13285
type: SIMPLE_ASSIGN
r_init[358] = r_min + 358.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13285};
  (data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (358.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13286
type: SIMPLE_ASSIGN
omega_c[358] = sqrt(G * Md / (r_init[358] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13286(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13286};
  modelica_real tmp1420;
  modelica_real tmp1421;
  modelica_real tmp1422;
  modelica_real tmp1423;
  modelica_real tmp1424;
  modelica_real tmp1425;
  modelica_real tmp1426;
  modelica_real tmp1427;
  modelica_real tmp1428;
  modelica_real tmp1429;
  tmp1420 = (data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */);
  tmp1421 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1422 = (tmp1420 * tmp1420) + (tmp1421 * tmp1421);
  tmp1423 = 1.5;
  if(tmp1422 < 0.0 && tmp1423 != 0.0)
  {
    tmp1425 = modf(tmp1423, &tmp1426);
    
    if(tmp1425 > 0.5)
    {
      tmp1425 -= 1.0;
      tmp1426 += 1.0;
    }
    else if(tmp1425 < -0.5)
    {
      tmp1425 += 1.0;
      tmp1426 -= 1.0;
    }
    
    if(fabs(tmp1425) < 1e-10)
      tmp1424 = pow(tmp1422, tmp1426);
    else
    {
      tmp1428 = modf(1.0/tmp1423, &tmp1427);
      if(tmp1428 > 0.5)
      {
        tmp1428 -= 1.0;
        tmp1427 += 1.0;
      }
      else if(tmp1428 < -0.5)
      {
        tmp1428 += 1.0;
        tmp1427 -= 1.0;
      }
      if(fabs(tmp1428) < 1e-10 && ((unsigned long)tmp1427 & 1))
      {
        tmp1424 = -pow(-tmp1422, tmp1425)*pow(tmp1422, tmp1426);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1422, tmp1423);
      }
    }
  }
  else
  {
    tmp1424 = pow(tmp1422, tmp1423);
  }
  if(isnan(tmp1424) || isinf(tmp1424))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1422, tmp1423);
  }tmp1429 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1424,"(r_init[358] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1429 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[358] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1429);
    }
  }
  (data->simulationInfo->realParameter[862] /* omega_c[358] PARAM */) = sqrt(tmp1429);
  TRACE_POP
}

/*
equation index: 13287
type: SIMPLE_ASSIGN
r_init[357] = r_min + 357.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13287};
  (data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (357.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13288
type: SIMPLE_ASSIGN
omega_c[357] = sqrt(G * Md / (r_init[357] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13288(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13288};
  modelica_real tmp1430;
  modelica_real tmp1431;
  modelica_real tmp1432;
  modelica_real tmp1433;
  modelica_real tmp1434;
  modelica_real tmp1435;
  modelica_real tmp1436;
  modelica_real tmp1437;
  modelica_real tmp1438;
  modelica_real tmp1439;
  tmp1430 = (data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */);
  tmp1431 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1432 = (tmp1430 * tmp1430) + (tmp1431 * tmp1431);
  tmp1433 = 1.5;
  if(tmp1432 < 0.0 && tmp1433 != 0.0)
  {
    tmp1435 = modf(tmp1433, &tmp1436);
    
    if(tmp1435 > 0.5)
    {
      tmp1435 -= 1.0;
      tmp1436 += 1.0;
    }
    else if(tmp1435 < -0.5)
    {
      tmp1435 += 1.0;
      tmp1436 -= 1.0;
    }
    
    if(fabs(tmp1435) < 1e-10)
      tmp1434 = pow(tmp1432, tmp1436);
    else
    {
      tmp1438 = modf(1.0/tmp1433, &tmp1437);
      if(tmp1438 > 0.5)
      {
        tmp1438 -= 1.0;
        tmp1437 += 1.0;
      }
      else if(tmp1438 < -0.5)
      {
        tmp1438 += 1.0;
        tmp1437 -= 1.0;
      }
      if(fabs(tmp1438) < 1e-10 && ((unsigned long)tmp1437 & 1))
      {
        tmp1434 = -pow(-tmp1432, tmp1435)*pow(tmp1432, tmp1436);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1432, tmp1433);
      }
    }
  }
  else
  {
    tmp1434 = pow(tmp1432, tmp1433);
  }
  if(isnan(tmp1434) || isinf(tmp1434))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1432, tmp1433);
  }tmp1439 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1434,"(r_init[357] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1439 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[357] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1439);
    }
  }
  (data->simulationInfo->realParameter[861] /* omega_c[357] PARAM */) = sqrt(tmp1439);
  TRACE_POP
}

/*
equation index: 13289
type: SIMPLE_ASSIGN
r_init[356] = r_min + 356.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13289};
  (data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (356.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13290
type: SIMPLE_ASSIGN
omega_c[356] = sqrt(G * Md / (r_init[356] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13290(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13290};
  modelica_real tmp1440;
  modelica_real tmp1441;
  modelica_real tmp1442;
  modelica_real tmp1443;
  modelica_real tmp1444;
  modelica_real tmp1445;
  modelica_real tmp1446;
  modelica_real tmp1447;
  modelica_real tmp1448;
  modelica_real tmp1449;
  tmp1440 = (data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */);
  tmp1441 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1442 = (tmp1440 * tmp1440) + (tmp1441 * tmp1441);
  tmp1443 = 1.5;
  if(tmp1442 < 0.0 && tmp1443 != 0.0)
  {
    tmp1445 = modf(tmp1443, &tmp1446);
    
    if(tmp1445 > 0.5)
    {
      tmp1445 -= 1.0;
      tmp1446 += 1.0;
    }
    else if(tmp1445 < -0.5)
    {
      tmp1445 += 1.0;
      tmp1446 -= 1.0;
    }
    
    if(fabs(tmp1445) < 1e-10)
      tmp1444 = pow(tmp1442, tmp1446);
    else
    {
      tmp1448 = modf(1.0/tmp1443, &tmp1447);
      if(tmp1448 > 0.5)
      {
        tmp1448 -= 1.0;
        tmp1447 += 1.0;
      }
      else if(tmp1448 < -0.5)
      {
        tmp1448 += 1.0;
        tmp1447 -= 1.0;
      }
      if(fabs(tmp1448) < 1e-10 && ((unsigned long)tmp1447 & 1))
      {
        tmp1444 = -pow(-tmp1442, tmp1445)*pow(tmp1442, tmp1446);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1442, tmp1443);
      }
    }
  }
  else
  {
    tmp1444 = pow(tmp1442, tmp1443);
  }
  if(isnan(tmp1444) || isinf(tmp1444))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1442, tmp1443);
  }tmp1449 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1444,"(r_init[356] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1449 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[356] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1449);
    }
  }
  (data->simulationInfo->realParameter[860] /* omega_c[356] PARAM */) = sqrt(tmp1449);
  TRACE_POP
}

/*
equation index: 13291
type: SIMPLE_ASSIGN
r_init[355] = r_min + 355.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13291};
  (data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (355.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13292
type: SIMPLE_ASSIGN
omega_c[355] = sqrt(G * Md / (r_init[355] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13292};
  modelica_real tmp1450;
  modelica_real tmp1451;
  modelica_real tmp1452;
  modelica_real tmp1453;
  modelica_real tmp1454;
  modelica_real tmp1455;
  modelica_real tmp1456;
  modelica_real tmp1457;
  modelica_real tmp1458;
  modelica_real tmp1459;
  tmp1450 = (data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */);
  tmp1451 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1452 = (tmp1450 * tmp1450) + (tmp1451 * tmp1451);
  tmp1453 = 1.5;
  if(tmp1452 < 0.0 && tmp1453 != 0.0)
  {
    tmp1455 = modf(tmp1453, &tmp1456);
    
    if(tmp1455 > 0.5)
    {
      tmp1455 -= 1.0;
      tmp1456 += 1.0;
    }
    else if(tmp1455 < -0.5)
    {
      tmp1455 += 1.0;
      tmp1456 -= 1.0;
    }
    
    if(fabs(tmp1455) < 1e-10)
      tmp1454 = pow(tmp1452, tmp1456);
    else
    {
      tmp1458 = modf(1.0/tmp1453, &tmp1457);
      if(tmp1458 > 0.5)
      {
        tmp1458 -= 1.0;
        tmp1457 += 1.0;
      }
      else if(tmp1458 < -0.5)
      {
        tmp1458 += 1.0;
        tmp1457 -= 1.0;
      }
      if(fabs(tmp1458) < 1e-10 && ((unsigned long)tmp1457 & 1))
      {
        tmp1454 = -pow(-tmp1452, tmp1455)*pow(tmp1452, tmp1456);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1452, tmp1453);
      }
    }
  }
  else
  {
    tmp1454 = pow(tmp1452, tmp1453);
  }
  if(isnan(tmp1454) || isinf(tmp1454))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1452, tmp1453);
  }tmp1459 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1454,"(r_init[355] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1459 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[355] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1459);
    }
  }
  (data->simulationInfo->realParameter[859] /* omega_c[355] PARAM */) = sqrt(tmp1459);
  TRACE_POP
}

/*
equation index: 13293
type: SIMPLE_ASSIGN
r_init[354] = r_min + 354.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13293};
  (data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (354.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13294
type: SIMPLE_ASSIGN
omega_c[354] = sqrt(G * Md / (r_init[354] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13294(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13294};
  modelica_real tmp1460;
  modelica_real tmp1461;
  modelica_real tmp1462;
  modelica_real tmp1463;
  modelica_real tmp1464;
  modelica_real tmp1465;
  modelica_real tmp1466;
  modelica_real tmp1467;
  modelica_real tmp1468;
  modelica_real tmp1469;
  tmp1460 = (data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */);
  tmp1461 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1462 = (tmp1460 * tmp1460) + (tmp1461 * tmp1461);
  tmp1463 = 1.5;
  if(tmp1462 < 0.0 && tmp1463 != 0.0)
  {
    tmp1465 = modf(tmp1463, &tmp1466);
    
    if(tmp1465 > 0.5)
    {
      tmp1465 -= 1.0;
      tmp1466 += 1.0;
    }
    else if(tmp1465 < -0.5)
    {
      tmp1465 += 1.0;
      tmp1466 -= 1.0;
    }
    
    if(fabs(tmp1465) < 1e-10)
      tmp1464 = pow(tmp1462, tmp1466);
    else
    {
      tmp1468 = modf(1.0/tmp1463, &tmp1467);
      if(tmp1468 > 0.5)
      {
        tmp1468 -= 1.0;
        tmp1467 += 1.0;
      }
      else if(tmp1468 < -0.5)
      {
        tmp1468 += 1.0;
        tmp1467 -= 1.0;
      }
      if(fabs(tmp1468) < 1e-10 && ((unsigned long)tmp1467 & 1))
      {
        tmp1464 = -pow(-tmp1462, tmp1465)*pow(tmp1462, tmp1466);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1462, tmp1463);
      }
    }
  }
  else
  {
    tmp1464 = pow(tmp1462, tmp1463);
  }
  if(isnan(tmp1464) || isinf(tmp1464))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1462, tmp1463);
  }tmp1469 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1464,"(r_init[354] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1469 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[354] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1469);
    }
  }
  (data->simulationInfo->realParameter[858] /* omega_c[354] PARAM */) = sqrt(tmp1469);
  TRACE_POP
}

/*
equation index: 13295
type: SIMPLE_ASSIGN
r_init[353] = r_min + 353.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13295};
  (data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (353.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13296
type: SIMPLE_ASSIGN
omega_c[353] = sqrt(G * Md / (r_init[353] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13296(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13296};
  modelica_real tmp1470;
  modelica_real tmp1471;
  modelica_real tmp1472;
  modelica_real tmp1473;
  modelica_real tmp1474;
  modelica_real tmp1475;
  modelica_real tmp1476;
  modelica_real tmp1477;
  modelica_real tmp1478;
  modelica_real tmp1479;
  tmp1470 = (data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */);
  tmp1471 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1472 = (tmp1470 * tmp1470) + (tmp1471 * tmp1471);
  tmp1473 = 1.5;
  if(tmp1472 < 0.0 && tmp1473 != 0.0)
  {
    tmp1475 = modf(tmp1473, &tmp1476);
    
    if(tmp1475 > 0.5)
    {
      tmp1475 -= 1.0;
      tmp1476 += 1.0;
    }
    else if(tmp1475 < -0.5)
    {
      tmp1475 += 1.0;
      tmp1476 -= 1.0;
    }
    
    if(fabs(tmp1475) < 1e-10)
      tmp1474 = pow(tmp1472, tmp1476);
    else
    {
      tmp1478 = modf(1.0/tmp1473, &tmp1477);
      if(tmp1478 > 0.5)
      {
        tmp1478 -= 1.0;
        tmp1477 += 1.0;
      }
      else if(tmp1478 < -0.5)
      {
        tmp1478 += 1.0;
        tmp1477 -= 1.0;
      }
      if(fabs(tmp1478) < 1e-10 && ((unsigned long)tmp1477 & 1))
      {
        tmp1474 = -pow(-tmp1472, tmp1475)*pow(tmp1472, tmp1476);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1472, tmp1473);
      }
    }
  }
  else
  {
    tmp1474 = pow(tmp1472, tmp1473);
  }
  if(isnan(tmp1474) || isinf(tmp1474))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1472, tmp1473);
  }tmp1479 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1474,"(r_init[353] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1479 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[353] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1479);
    }
  }
  (data->simulationInfo->realParameter[857] /* omega_c[353] PARAM */) = sqrt(tmp1479);
  TRACE_POP
}

/*
equation index: 13297
type: SIMPLE_ASSIGN
r_init[352] = r_min + 352.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13297};
  (data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (352.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13298
type: SIMPLE_ASSIGN
omega_c[352] = sqrt(G * Md / (r_init[352] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13298(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13298};
  modelica_real tmp1480;
  modelica_real tmp1481;
  modelica_real tmp1482;
  modelica_real tmp1483;
  modelica_real tmp1484;
  modelica_real tmp1485;
  modelica_real tmp1486;
  modelica_real tmp1487;
  modelica_real tmp1488;
  modelica_real tmp1489;
  tmp1480 = (data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */);
  tmp1481 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1482 = (tmp1480 * tmp1480) + (tmp1481 * tmp1481);
  tmp1483 = 1.5;
  if(tmp1482 < 0.0 && tmp1483 != 0.0)
  {
    tmp1485 = modf(tmp1483, &tmp1486);
    
    if(tmp1485 > 0.5)
    {
      tmp1485 -= 1.0;
      tmp1486 += 1.0;
    }
    else if(tmp1485 < -0.5)
    {
      tmp1485 += 1.0;
      tmp1486 -= 1.0;
    }
    
    if(fabs(tmp1485) < 1e-10)
      tmp1484 = pow(tmp1482, tmp1486);
    else
    {
      tmp1488 = modf(1.0/tmp1483, &tmp1487);
      if(tmp1488 > 0.5)
      {
        tmp1488 -= 1.0;
        tmp1487 += 1.0;
      }
      else if(tmp1488 < -0.5)
      {
        tmp1488 += 1.0;
        tmp1487 -= 1.0;
      }
      if(fabs(tmp1488) < 1e-10 && ((unsigned long)tmp1487 & 1))
      {
        tmp1484 = -pow(-tmp1482, tmp1485)*pow(tmp1482, tmp1486);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1482, tmp1483);
      }
    }
  }
  else
  {
    tmp1484 = pow(tmp1482, tmp1483);
  }
  if(isnan(tmp1484) || isinf(tmp1484))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1482, tmp1483);
  }tmp1489 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1484,"(r_init[352] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1489 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[352] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1489);
    }
  }
  (data->simulationInfo->realParameter[856] /* omega_c[352] PARAM */) = sqrt(tmp1489);
  TRACE_POP
}

/*
equation index: 13299
type: SIMPLE_ASSIGN
r_init[351] = r_min + 351.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13299};
  (data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (351.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13300
type: SIMPLE_ASSIGN
omega_c[351] = sqrt(G * Md / (r_init[351] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13300};
  modelica_real tmp1490;
  modelica_real tmp1491;
  modelica_real tmp1492;
  modelica_real tmp1493;
  modelica_real tmp1494;
  modelica_real tmp1495;
  modelica_real tmp1496;
  modelica_real tmp1497;
  modelica_real tmp1498;
  modelica_real tmp1499;
  tmp1490 = (data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */);
  tmp1491 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1492 = (tmp1490 * tmp1490) + (tmp1491 * tmp1491);
  tmp1493 = 1.5;
  if(tmp1492 < 0.0 && tmp1493 != 0.0)
  {
    tmp1495 = modf(tmp1493, &tmp1496);
    
    if(tmp1495 > 0.5)
    {
      tmp1495 -= 1.0;
      tmp1496 += 1.0;
    }
    else if(tmp1495 < -0.5)
    {
      tmp1495 += 1.0;
      tmp1496 -= 1.0;
    }
    
    if(fabs(tmp1495) < 1e-10)
      tmp1494 = pow(tmp1492, tmp1496);
    else
    {
      tmp1498 = modf(1.0/tmp1493, &tmp1497);
      if(tmp1498 > 0.5)
      {
        tmp1498 -= 1.0;
        tmp1497 += 1.0;
      }
      else if(tmp1498 < -0.5)
      {
        tmp1498 += 1.0;
        tmp1497 -= 1.0;
      }
      if(fabs(tmp1498) < 1e-10 && ((unsigned long)tmp1497 & 1))
      {
        tmp1494 = -pow(-tmp1492, tmp1495)*pow(tmp1492, tmp1496);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1492, tmp1493);
      }
    }
  }
  else
  {
    tmp1494 = pow(tmp1492, tmp1493);
  }
  if(isnan(tmp1494) || isinf(tmp1494))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1492, tmp1493);
  }tmp1499 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1494,"(r_init[351] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1499 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[351] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1499);
    }
  }
  (data->simulationInfo->realParameter[855] /* omega_c[351] PARAM */) = sqrt(tmp1499);
  TRACE_POP
}

/*
equation index: 13301
type: SIMPLE_ASSIGN
r_init[350] = r_min + 350.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13301};
  (data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (350.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13302
type: SIMPLE_ASSIGN
omega_c[350] = sqrt(G * Md / (r_init[350] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13302(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13302};
  modelica_real tmp1500;
  modelica_real tmp1501;
  modelica_real tmp1502;
  modelica_real tmp1503;
  modelica_real tmp1504;
  modelica_real tmp1505;
  modelica_real tmp1506;
  modelica_real tmp1507;
  modelica_real tmp1508;
  modelica_real tmp1509;
  tmp1500 = (data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */);
  tmp1501 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1502 = (tmp1500 * tmp1500) + (tmp1501 * tmp1501);
  tmp1503 = 1.5;
  if(tmp1502 < 0.0 && tmp1503 != 0.0)
  {
    tmp1505 = modf(tmp1503, &tmp1506);
    
    if(tmp1505 > 0.5)
    {
      tmp1505 -= 1.0;
      tmp1506 += 1.0;
    }
    else if(tmp1505 < -0.5)
    {
      tmp1505 += 1.0;
      tmp1506 -= 1.0;
    }
    
    if(fabs(tmp1505) < 1e-10)
      tmp1504 = pow(tmp1502, tmp1506);
    else
    {
      tmp1508 = modf(1.0/tmp1503, &tmp1507);
      if(tmp1508 > 0.5)
      {
        tmp1508 -= 1.0;
        tmp1507 += 1.0;
      }
      else if(tmp1508 < -0.5)
      {
        tmp1508 += 1.0;
        tmp1507 -= 1.0;
      }
      if(fabs(tmp1508) < 1e-10 && ((unsigned long)tmp1507 & 1))
      {
        tmp1504 = -pow(-tmp1502, tmp1505)*pow(tmp1502, tmp1506);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1502, tmp1503);
      }
    }
  }
  else
  {
    tmp1504 = pow(tmp1502, tmp1503);
  }
  if(isnan(tmp1504) || isinf(tmp1504))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1502, tmp1503);
  }tmp1509 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1504,"(r_init[350] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1509 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[350] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1509);
    }
  }
  (data->simulationInfo->realParameter[854] /* omega_c[350] PARAM */) = sqrt(tmp1509);
  TRACE_POP
}

/*
equation index: 13303
type: SIMPLE_ASSIGN
r_init[349] = r_min + 349.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13303};
  (data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (349.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13304
type: SIMPLE_ASSIGN
omega_c[349] = sqrt(G * Md / (r_init[349] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13304(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13304};
  modelica_real tmp1510;
  modelica_real tmp1511;
  modelica_real tmp1512;
  modelica_real tmp1513;
  modelica_real tmp1514;
  modelica_real tmp1515;
  modelica_real tmp1516;
  modelica_real tmp1517;
  modelica_real tmp1518;
  modelica_real tmp1519;
  tmp1510 = (data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */);
  tmp1511 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1512 = (tmp1510 * tmp1510) + (tmp1511 * tmp1511);
  tmp1513 = 1.5;
  if(tmp1512 < 0.0 && tmp1513 != 0.0)
  {
    tmp1515 = modf(tmp1513, &tmp1516);
    
    if(tmp1515 > 0.5)
    {
      tmp1515 -= 1.0;
      tmp1516 += 1.0;
    }
    else if(tmp1515 < -0.5)
    {
      tmp1515 += 1.0;
      tmp1516 -= 1.0;
    }
    
    if(fabs(tmp1515) < 1e-10)
      tmp1514 = pow(tmp1512, tmp1516);
    else
    {
      tmp1518 = modf(1.0/tmp1513, &tmp1517);
      if(tmp1518 > 0.5)
      {
        tmp1518 -= 1.0;
        tmp1517 += 1.0;
      }
      else if(tmp1518 < -0.5)
      {
        tmp1518 += 1.0;
        tmp1517 -= 1.0;
      }
      if(fabs(tmp1518) < 1e-10 && ((unsigned long)tmp1517 & 1))
      {
        tmp1514 = -pow(-tmp1512, tmp1515)*pow(tmp1512, tmp1516);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1512, tmp1513);
      }
    }
  }
  else
  {
    tmp1514 = pow(tmp1512, tmp1513);
  }
  if(isnan(tmp1514) || isinf(tmp1514))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1512, tmp1513);
  }tmp1519 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1514,"(r_init[349] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1519 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[349] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1519);
    }
  }
  (data->simulationInfo->realParameter[853] /* omega_c[349] PARAM */) = sqrt(tmp1519);
  TRACE_POP
}

/*
equation index: 13305
type: SIMPLE_ASSIGN
r_init[348] = r_min + 348.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13305};
  (data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (348.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13306
type: SIMPLE_ASSIGN
omega_c[348] = sqrt(G * Md / (r_init[348] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13306(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13306};
  modelica_real tmp1520;
  modelica_real tmp1521;
  modelica_real tmp1522;
  modelica_real tmp1523;
  modelica_real tmp1524;
  modelica_real tmp1525;
  modelica_real tmp1526;
  modelica_real tmp1527;
  modelica_real tmp1528;
  modelica_real tmp1529;
  tmp1520 = (data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */);
  tmp1521 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1522 = (tmp1520 * tmp1520) + (tmp1521 * tmp1521);
  tmp1523 = 1.5;
  if(tmp1522 < 0.0 && tmp1523 != 0.0)
  {
    tmp1525 = modf(tmp1523, &tmp1526);
    
    if(tmp1525 > 0.5)
    {
      tmp1525 -= 1.0;
      tmp1526 += 1.0;
    }
    else if(tmp1525 < -0.5)
    {
      tmp1525 += 1.0;
      tmp1526 -= 1.0;
    }
    
    if(fabs(tmp1525) < 1e-10)
      tmp1524 = pow(tmp1522, tmp1526);
    else
    {
      tmp1528 = modf(1.0/tmp1523, &tmp1527);
      if(tmp1528 > 0.5)
      {
        tmp1528 -= 1.0;
        tmp1527 += 1.0;
      }
      else if(tmp1528 < -0.5)
      {
        tmp1528 += 1.0;
        tmp1527 -= 1.0;
      }
      if(fabs(tmp1528) < 1e-10 && ((unsigned long)tmp1527 & 1))
      {
        tmp1524 = -pow(-tmp1522, tmp1525)*pow(tmp1522, tmp1526);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1522, tmp1523);
      }
    }
  }
  else
  {
    tmp1524 = pow(tmp1522, tmp1523);
  }
  if(isnan(tmp1524) || isinf(tmp1524))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1522, tmp1523);
  }tmp1529 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1524,"(r_init[348] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1529 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[348] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1529);
    }
  }
  (data->simulationInfo->realParameter[852] /* omega_c[348] PARAM */) = sqrt(tmp1529);
  TRACE_POP
}

/*
equation index: 13307
type: SIMPLE_ASSIGN
r_init[347] = r_min + 347.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13307};
  (data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (347.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13308
type: SIMPLE_ASSIGN
omega_c[347] = sqrt(G * Md / (r_init[347] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13308};
  modelica_real tmp1530;
  modelica_real tmp1531;
  modelica_real tmp1532;
  modelica_real tmp1533;
  modelica_real tmp1534;
  modelica_real tmp1535;
  modelica_real tmp1536;
  modelica_real tmp1537;
  modelica_real tmp1538;
  modelica_real tmp1539;
  tmp1530 = (data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */);
  tmp1531 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1532 = (tmp1530 * tmp1530) + (tmp1531 * tmp1531);
  tmp1533 = 1.5;
  if(tmp1532 < 0.0 && tmp1533 != 0.0)
  {
    tmp1535 = modf(tmp1533, &tmp1536);
    
    if(tmp1535 > 0.5)
    {
      tmp1535 -= 1.0;
      tmp1536 += 1.0;
    }
    else if(tmp1535 < -0.5)
    {
      tmp1535 += 1.0;
      tmp1536 -= 1.0;
    }
    
    if(fabs(tmp1535) < 1e-10)
      tmp1534 = pow(tmp1532, tmp1536);
    else
    {
      tmp1538 = modf(1.0/tmp1533, &tmp1537);
      if(tmp1538 > 0.5)
      {
        tmp1538 -= 1.0;
        tmp1537 += 1.0;
      }
      else if(tmp1538 < -0.5)
      {
        tmp1538 += 1.0;
        tmp1537 -= 1.0;
      }
      if(fabs(tmp1538) < 1e-10 && ((unsigned long)tmp1537 & 1))
      {
        tmp1534 = -pow(-tmp1532, tmp1535)*pow(tmp1532, tmp1536);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1532, tmp1533);
      }
    }
  }
  else
  {
    tmp1534 = pow(tmp1532, tmp1533);
  }
  if(isnan(tmp1534) || isinf(tmp1534))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1532, tmp1533);
  }tmp1539 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1534,"(r_init[347] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1539 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[347] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1539);
    }
  }
  (data->simulationInfo->realParameter[851] /* omega_c[347] PARAM */) = sqrt(tmp1539);
  TRACE_POP
}

/*
equation index: 13309
type: SIMPLE_ASSIGN
r_init[346] = r_min + 346.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13309};
  (data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (346.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13310
type: SIMPLE_ASSIGN
omega_c[346] = sqrt(G * Md / (r_init[346] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13310(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13310};
  modelica_real tmp1540;
  modelica_real tmp1541;
  modelica_real tmp1542;
  modelica_real tmp1543;
  modelica_real tmp1544;
  modelica_real tmp1545;
  modelica_real tmp1546;
  modelica_real tmp1547;
  modelica_real tmp1548;
  modelica_real tmp1549;
  tmp1540 = (data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */);
  tmp1541 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1542 = (tmp1540 * tmp1540) + (tmp1541 * tmp1541);
  tmp1543 = 1.5;
  if(tmp1542 < 0.0 && tmp1543 != 0.0)
  {
    tmp1545 = modf(tmp1543, &tmp1546);
    
    if(tmp1545 > 0.5)
    {
      tmp1545 -= 1.0;
      tmp1546 += 1.0;
    }
    else if(tmp1545 < -0.5)
    {
      tmp1545 += 1.0;
      tmp1546 -= 1.0;
    }
    
    if(fabs(tmp1545) < 1e-10)
      tmp1544 = pow(tmp1542, tmp1546);
    else
    {
      tmp1548 = modf(1.0/tmp1543, &tmp1547);
      if(tmp1548 > 0.5)
      {
        tmp1548 -= 1.0;
        tmp1547 += 1.0;
      }
      else if(tmp1548 < -0.5)
      {
        tmp1548 += 1.0;
        tmp1547 -= 1.0;
      }
      if(fabs(tmp1548) < 1e-10 && ((unsigned long)tmp1547 & 1))
      {
        tmp1544 = -pow(-tmp1542, tmp1545)*pow(tmp1542, tmp1546);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1542, tmp1543);
      }
    }
  }
  else
  {
    tmp1544 = pow(tmp1542, tmp1543);
  }
  if(isnan(tmp1544) || isinf(tmp1544))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1542, tmp1543);
  }tmp1549 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1544,"(r_init[346] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1549 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[346] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1549);
    }
  }
  (data->simulationInfo->realParameter[850] /* omega_c[346] PARAM */) = sqrt(tmp1549);
  TRACE_POP
}

/*
equation index: 13311
type: SIMPLE_ASSIGN
r_init[345] = r_min + 345.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13311};
  (data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (345.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13312
type: SIMPLE_ASSIGN
omega_c[345] = sqrt(G * Md / (r_init[345] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13312(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13312};
  modelica_real tmp1550;
  modelica_real tmp1551;
  modelica_real tmp1552;
  modelica_real tmp1553;
  modelica_real tmp1554;
  modelica_real tmp1555;
  modelica_real tmp1556;
  modelica_real tmp1557;
  modelica_real tmp1558;
  modelica_real tmp1559;
  tmp1550 = (data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */);
  tmp1551 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1552 = (tmp1550 * tmp1550) + (tmp1551 * tmp1551);
  tmp1553 = 1.5;
  if(tmp1552 < 0.0 && tmp1553 != 0.0)
  {
    tmp1555 = modf(tmp1553, &tmp1556);
    
    if(tmp1555 > 0.5)
    {
      tmp1555 -= 1.0;
      tmp1556 += 1.0;
    }
    else if(tmp1555 < -0.5)
    {
      tmp1555 += 1.0;
      tmp1556 -= 1.0;
    }
    
    if(fabs(tmp1555) < 1e-10)
      tmp1554 = pow(tmp1552, tmp1556);
    else
    {
      tmp1558 = modf(1.0/tmp1553, &tmp1557);
      if(tmp1558 > 0.5)
      {
        tmp1558 -= 1.0;
        tmp1557 += 1.0;
      }
      else if(tmp1558 < -0.5)
      {
        tmp1558 += 1.0;
        tmp1557 -= 1.0;
      }
      if(fabs(tmp1558) < 1e-10 && ((unsigned long)tmp1557 & 1))
      {
        tmp1554 = -pow(-tmp1552, tmp1555)*pow(tmp1552, tmp1556);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1552, tmp1553);
      }
    }
  }
  else
  {
    tmp1554 = pow(tmp1552, tmp1553);
  }
  if(isnan(tmp1554) || isinf(tmp1554))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1552, tmp1553);
  }tmp1559 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1554,"(r_init[345] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1559 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[345] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1559);
    }
  }
  (data->simulationInfo->realParameter[849] /* omega_c[345] PARAM */) = sqrt(tmp1559);
  TRACE_POP
}

/*
equation index: 13313
type: SIMPLE_ASSIGN
r_init[344] = r_min + 344.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13313};
  (data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (344.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13314
type: SIMPLE_ASSIGN
omega_c[344] = sqrt(G * Md / (r_init[344] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13314(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13314};
  modelica_real tmp1560;
  modelica_real tmp1561;
  modelica_real tmp1562;
  modelica_real tmp1563;
  modelica_real tmp1564;
  modelica_real tmp1565;
  modelica_real tmp1566;
  modelica_real tmp1567;
  modelica_real tmp1568;
  modelica_real tmp1569;
  tmp1560 = (data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */);
  tmp1561 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1562 = (tmp1560 * tmp1560) + (tmp1561 * tmp1561);
  tmp1563 = 1.5;
  if(tmp1562 < 0.0 && tmp1563 != 0.0)
  {
    tmp1565 = modf(tmp1563, &tmp1566);
    
    if(tmp1565 > 0.5)
    {
      tmp1565 -= 1.0;
      tmp1566 += 1.0;
    }
    else if(tmp1565 < -0.5)
    {
      tmp1565 += 1.0;
      tmp1566 -= 1.0;
    }
    
    if(fabs(tmp1565) < 1e-10)
      tmp1564 = pow(tmp1562, tmp1566);
    else
    {
      tmp1568 = modf(1.0/tmp1563, &tmp1567);
      if(tmp1568 > 0.5)
      {
        tmp1568 -= 1.0;
        tmp1567 += 1.0;
      }
      else if(tmp1568 < -0.5)
      {
        tmp1568 += 1.0;
        tmp1567 -= 1.0;
      }
      if(fabs(tmp1568) < 1e-10 && ((unsigned long)tmp1567 & 1))
      {
        tmp1564 = -pow(-tmp1562, tmp1565)*pow(tmp1562, tmp1566);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1562, tmp1563);
      }
    }
  }
  else
  {
    tmp1564 = pow(tmp1562, tmp1563);
  }
  if(isnan(tmp1564) || isinf(tmp1564))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1562, tmp1563);
  }tmp1569 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1564,"(r_init[344] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1569 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[344] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1569);
    }
  }
  (data->simulationInfo->realParameter[848] /* omega_c[344] PARAM */) = sqrt(tmp1569);
  TRACE_POP
}

/*
equation index: 13315
type: SIMPLE_ASSIGN
r_init[343] = r_min + 343.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13315};
  (data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (343.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13316
type: SIMPLE_ASSIGN
omega_c[343] = sqrt(G * Md / (r_init[343] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13316};
  modelica_real tmp1570;
  modelica_real tmp1571;
  modelica_real tmp1572;
  modelica_real tmp1573;
  modelica_real tmp1574;
  modelica_real tmp1575;
  modelica_real tmp1576;
  modelica_real tmp1577;
  modelica_real tmp1578;
  modelica_real tmp1579;
  tmp1570 = (data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */);
  tmp1571 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1572 = (tmp1570 * tmp1570) + (tmp1571 * tmp1571);
  tmp1573 = 1.5;
  if(tmp1572 < 0.0 && tmp1573 != 0.0)
  {
    tmp1575 = modf(tmp1573, &tmp1576);
    
    if(tmp1575 > 0.5)
    {
      tmp1575 -= 1.0;
      tmp1576 += 1.0;
    }
    else if(tmp1575 < -0.5)
    {
      tmp1575 += 1.0;
      tmp1576 -= 1.0;
    }
    
    if(fabs(tmp1575) < 1e-10)
      tmp1574 = pow(tmp1572, tmp1576);
    else
    {
      tmp1578 = modf(1.0/tmp1573, &tmp1577);
      if(tmp1578 > 0.5)
      {
        tmp1578 -= 1.0;
        tmp1577 += 1.0;
      }
      else if(tmp1578 < -0.5)
      {
        tmp1578 += 1.0;
        tmp1577 -= 1.0;
      }
      if(fabs(tmp1578) < 1e-10 && ((unsigned long)tmp1577 & 1))
      {
        tmp1574 = -pow(-tmp1572, tmp1575)*pow(tmp1572, tmp1576);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1572, tmp1573);
      }
    }
  }
  else
  {
    tmp1574 = pow(tmp1572, tmp1573);
  }
  if(isnan(tmp1574) || isinf(tmp1574))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1572, tmp1573);
  }tmp1579 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1574,"(r_init[343] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1579 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[343] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1579);
    }
  }
  (data->simulationInfo->realParameter[847] /* omega_c[343] PARAM */) = sqrt(tmp1579);
  TRACE_POP
}

/*
equation index: 13317
type: SIMPLE_ASSIGN
r_init[342] = r_min + 342.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13317};
  (data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (342.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13318
type: SIMPLE_ASSIGN
omega_c[342] = sqrt(G * Md / (r_init[342] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13318(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13318};
  modelica_real tmp1580;
  modelica_real tmp1581;
  modelica_real tmp1582;
  modelica_real tmp1583;
  modelica_real tmp1584;
  modelica_real tmp1585;
  modelica_real tmp1586;
  modelica_real tmp1587;
  modelica_real tmp1588;
  modelica_real tmp1589;
  tmp1580 = (data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */);
  tmp1581 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1582 = (tmp1580 * tmp1580) + (tmp1581 * tmp1581);
  tmp1583 = 1.5;
  if(tmp1582 < 0.0 && tmp1583 != 0.0)
  {
    tmp1585 = modf(tmp1583, &tmp1586);
    
    if(tmp1585 > 0.5)
    {
      tmp1585 -= 1.0;
      tmp1586 += 1.0;
    }
    else if(tmp1585 < -0.5)
    {
      tmp1585 += 1.0;
      tmp1586 -= 1.0;
    }
    
    if(fabs(tmp1585) < 1e-10)
      tmp1584 = pow(tmp1582, tmp1586);
    else
    {
      tmp1588 = modf(1.0/tmp1583, &tmp1587);
      if(tmp1588 > 0.5)
      {
        tmp1588 -= 1.0;
        tmp1587 += 1.0;
      }
      else if(tmp1588 < -0.5)
      {
        tmp1588 += 1.0;
        tmp1587 -= 1.0;
      }
      if(fabs(tmp1588) < 1e-10 && ((unsigned long)tmp1587 & 1))
      {
        tmp1584 = -pow(-tmp1582, tmp1585)*pow(tmp1582, tmp1586);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1582, tmp1583);
      }
    }
  }
  else
  {
    tmp1584 = pow(tmp1582, tmp1583);
  }
  if(isnan(tmp1584) || isinf(tmp1584))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1582, tmp1583);
  }tmp1589 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1584,"(r_init[342] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1589 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[342] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1589);
    }
  }
  (data->simulationInfo->realParameter[846] /* omega_c[342] PARAM */) = sqrt(tmp1589);
  TRACE_POP
}

/*
equation index: 13319
type: SIMPLE_ASSIGN
r_init[341] = r_min + 341.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13319};
  (data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (341.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13320
type: SIMPLE_ASSIGN
omega_c[341] = sqrt(G * Md / (r_init[341] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13320(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13320};
  modelica_real tmp1590;
  modelica_real tmp1591;
  modelica_real tmp1592;
  modelica_real tmp1593;
  modelica_real tmp1594;
  modelica_real tmp1595;
  modelica_real tmp1596;
  modelica_real tmp1597;
  modelica_real tmp1598;
  modelica_real tmp1599;
  tmp1590 = (data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */);
  tmp1591 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1592 = (tmp1590 * tmp1590) + (tmp1591 * tmp1591);
  tmp1593 = 1.5;
  if(tmp1592 < 0.0 && tmp1593 != 0.0)
  {
    tmp1595 = modf(tmp1593, &tmp1596);
    
    if(tmp1595 > 0.5)
    {
      tmp1595 -= 1.0;
      tmp1596 += 1.0;
    }
    else if(tmp1595 < -0.5)
    {
      tmp1595 += 1.0;
      tmp1596 -= 1.0;
    }
    
    if(fabs(tmp1595) < 1e-10)
      tmp1594 = pow(tmp1592, tmp1596);
    else
    {
      tmp1598 = modf(1.0/tmp1593, &tmp1597);
      if(tmp1598 > 0.5)
      {
        tmp1598 -= 1.0;
        tmp1597 += 1.0;
      }
      else if(tmp1598 < -0.5)
      {
        tmp1598 += 1.0;
        tmp1597 -= 1.0;
      }
      if(fabs(tmp1598) < 1e-10 && ((unsigned long)tmp1597 & 1))
      {
        tmp1594 = -pow(-tmp1592, tmp1595)*pow(tmp1592, tmp1596);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1592, tmp1593);
      }
    }
  }
  else
  {
    tmp1594 = pow(tmp1592, tmp1593);
  }
  if(isnan(tmp1594) || isinf(tmp1594))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1592, tmp1593);
  }tmp1599 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1594,"(r_init[341] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1599 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[341] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1599);
    }
  }
  (data->simulationInfo->realParameter[845] /* omega_c[341] PARAM */) = sqrt(tmp1599);
  TRACE_POP
}

/*
equation index: 13321
type: SIMPLE_ASSIGN
r_init[340] = r_min + 340.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13321};
  (data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (340.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13322
type: SIMPLE_ASSIGN
omega_c[340] = sqrt(G * Md / (r_init[340] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13322(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13322};
  modelica_real tmp1600;
  modelica_real tmp1601;
  modelica_real tmp1602;
  modelica_real tmp1603;
  modelica_real tmp1604;
  modelica_real tmp1605;
  modelica_real tmp1606;
  modelica_real tmp1607;
  modelica_real tmp1608;
  modelica_real tmp1609;
  tmp1600 = (data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */);
  tmp1601 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1602 = (tmp1600 * tmp1600) + (tmp1601 * tmp1601);
  tmp1603 = 1.5;
  if(tmp1602 < 0.0 && tmp1603 != 0.0)
  {
    tmp1605 = modf(tmp1603, &tmp1606);
    
    if(tmp1605 > 0.5)
    {
      tmp1605 -= 1.0;
      tmp1606 += 1.0;
    }
    else if(tmp1605 < -0.5)
    {
      tmp1605 += 1.0;
      tmp1606 -= 1.0;
    }
    
    if(fabs(tmp1605) < 1e-10)
      tmp1604 = pow(tmp1602, tmp1606);
    else
    {
      tmp1608 = modf(1.0/tmp1603, &tmp1607);
      if(tmp1608 > 0.5)
      {
        tmp1608 -= 1.0;
        tmp1607 += 1.0;
      }
      else if(tmp1608 < -0.5)
      {
        tmp1608 += 1.0;
        tmp1607 -= 1.0;
      }
      if(fabs(tmp1608) < 1e-10 && ((unsigned long)tmp1607 & 1))
      {
        tmp1604 = -pow(-tmp1602, tmp1605)*pow(tmp1602, tmp1606);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1602, tmp1603);
      }
    }
  }
  else
  {
    tmp1604 = pow(tmp1602, tmp1603);
  }
  if(isnan(tmp1604) || isinf(tmp1604))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1602, tmp1603);
  }tmp1609 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1604,"(r_init[340] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1609 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[340] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1609);
    }
  }
  (data->simulationInfo->realParameter[844] /* omega_c[340] PARAM */) = sqrt(tmp1609);
  TRACE_POP
}

/*
equation index: 13323
type: SIMPLE_ASSIGN
r_init[339] = r_min + 339.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13323};
  (data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (339.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13324
type: SIMPLE_ASSIGN
omega_c[339] = sqrt(G * Md / (r_init[339] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13324};
  modelica_real tmp1610;
  modelica_real tmp1611;
  modelica_real tmp1612;
  modelica_real tmp1613;
  modelica_real tmp1614;
  modelica_real tmp1615;
  modelica_real tmp1616;
  modelica_real tmp1617;
  modelica_real tmp1618;
  modelica_real tmp1619;
  tmp1610 = (data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */);
  tmp1611 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1612 = (tmp1610 * tmp1610) + (tmp1611 * tmp1611);
  tmp1613 = 1.5;
  if(tmp1612 < 0.0 && tmp1613 != 0.0)
  {
    tmp1615 = modf(tmp1613, &tmp1616);
    
    if(tmp1615 > 0.5)
    {
      tmp1615 -= 1.0;
      tmp1616 += 1.0;
    }
    else if(tmp1615 < -0.5)
    {
      tmp1615 += 1.0;
      tmp1616 -= 1.0;
    }
    
    if(fabs(tmp1615) < 1e-10)
      tmp1614 = pow(tmp1612, tmp1616);
    else
    {
      tmp1618 = modf(1.0/tmp1613, &tmp1617);
      if(tmp1618 > 0.5)
      {
        tmp1618 -= 1.0;
        tmp1617 += 1.0;
      }
      else if(tmp1618 < -0.5)
      {
        tmp1618 += 1.0;
        tmp1617 -= 1.0;
      }
      if(fabs(tmp1618) < 1e-10 && ((unsigned long)tmp1617 & 1))
      {
        tmp1614 = -pow(-tmp1612, tmp1615)*pow(tmp1612, tmp1616);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1612, tmp1613);
      }
    }
  }
  else
  {
    tmp1614 = pow(tmp1612, tmp1613);
  }
  if(isnan(tmp1614) || isinf(tmp1614))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1612, tmp1613);
  }tmp1619 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1614,"(r_init[339] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1619 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[339] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1619);
    }
  }
  (data->simulationInfo->realParameter[843] /* omega_c[339] PARAM */) = sqrt(tmp1619);
  TRACE_POP
}

/*
equation index: 13325
type: SIMPLE_ASSIGN
r_init[338] = r_min + 338.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13325};
  (data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (338.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13326
type: SIMPLE_ASSIGN
omega_c[338] = sqrt(G * Md / (r_init[338] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13326(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13326};
  modelica_real tmp1620;
  modelica_real tmp1621;
  modelica_real tmp1622;
  modelica_real tmp1623;
  modelica_real tmp1624;
  modelica_real tmp1625;
  modelica_real tmp1626;
  modelica_real tmp1627;
  modelica_real tmp1628;
  modelica_real tmp1629;
  tmp1620 = (data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */);
  tmp1621 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1622 = (tmp1620 * tmp1620) + (tmp1621 * tmp1621);
  tmp1623 = 1.5;
  if(tmp1622 < 0.0 && tmp1623 != 0.0)
  {
    tmp1625 = modf(tmp1623, &tmp1626);
    
    if(tmp1625 > 0.5)
    {
      tmp1625 -= 1.0;
      tmp1626 += 1.0;
    }
    else if(tmp1625 < -0.5)
    {
      tmp1625 += 1.0;
      tmp1626 -= 1.0;
    }
    
    if(fabs(tmp1625) < 1e-10)
      tmp1624 = pow(tmp1622, tmp1626);
    else
    {
      tmp1628 = modf(1.0/tmp1623, &tmp1627);
      if(tmp1628 > 0.5)
      {
        tmp1628 -= 1.0;
        tmp1627 += 1.0;
      }
      else if(tmp1628 < -0.5)
      {
        tmp1628 += 1.0;
        tmp1627 -= 1.0;
      }
      if(fabs(tmp1628) < 1e-10 && ((unsigned long)tmp1627 & 1))
      {
        tmp1624 = -pow(-tmp1622, tmp1625)*pow(tmp1622, tmp1626);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1622, tmp1623);
      }
    }
  }
  else
  {
    tmp1624 = pow(tmp1622, tmp1623);
  }
  if(isnan(tmp1624) || isinf(tmp1624))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1622, tmp1623);
  }tmp1629 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1624,"(r_init[338] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1629 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[338] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1629);
    }
  }
  (data->simulationInfo->realParameter[842] /* omega_c[338] PARAM */) = sqrt(tmp1629);
  TRACE_POP
}

/*
equation index: 13327
type: SIMPLE_ASSIGN
r_init[337] = r_min + 337.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13327};
  (data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (337.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13328
type: SIMPLE_ASSIGN
omega_c[337] = sqrt(G * Md / (r_init[337] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13328(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13328};
  modelica_real tmp1630;
  modelica_real tmp1631;
  modelica_real tmp1632;
  modelica_real tmp1633;
  modelica_real tmp1634;
  modelica_real tmp1635;
  modelica_real tmp1636;
  modelica_real tmp1637;
  modelica_real tmp1638;
  modelica_real tmp1639;
  tmp1630 = (data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */);
  tmp1631 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1632 = (tmp1630 * tmp1630) + (tmp1631 * tmp1631);
  tmp1633 = 1.5;
  if(tmp1632 < 0.0 && tmp1633 != 0.0)
  {
    tmp1635 = modf(tmp1633, &tmp1636);
    
    if(tmp1635 > 0.5)
    {
      tmp1635 -= 1.0;
      tmp1636 += 1.0;
    }
    else if(tmp1635 < -0.5)
    {
      tmp1635 += 1.0;
      tmp1636 -= 1.0;
    }
    
    if(fabs(tmp1635) < 1e-10)
      tmp1634 = pow(tmp1632, tmp1636);
    else
    {
      tmp1638 = modf(1.0/tmp1633, &tmp1637);
      if(tmp1638 > 0.5)
      {
        tmp1638 -= 1.0;
        tmp1637 += 1.0;
      }
      else if(tmp1638 < -0.5)
      {
        tmp1638 += 1.0;
        tmp1637 -= 1.0;
      }
      if(fabs(tmp1638) < 1e-10 && ((unsigned long)tmp1637 & 1))
      {
        tmp1634 = -pow(-tmp1632, tmp1635)*pow(tmp1632, tmp1636);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1632, tmp1633);
      }
    }
  }
  else
  {
    tmp1634 = pow(tmp1632, tmp1633);
  }
  if(isnan(tmp1634) || isinf(tmp1634))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1632, tmp1633);
  }tmp1639 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1634,"(r_init[337] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1639 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[337] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1639);
    }
  }
  (data->simulationInfo->realParameter[841] /* omega_c[337] PARAM */) = sqrt(tmp1639);
  TRACE_POP
}

/*
equation index: 13329
type: SIMPLE_ASSIGN
r_init[336] = r_min + 336.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13329};
  (data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (336.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13330
type: SIMPLE_ASSIGN
omega_c[336] = sqrt(G * Md / (r_init[336] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13330(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13330};
  modelica_real tmp1640;
  modelica_real tmp1641;
  modelica_real tmp1642;
  modelica_real tmp1643;
  modelica_real tmp1644;
  modelica_real tmp1645;
  modelica_real tmp1646;
  modelica_real tmp1647;
  modelica_real tmp1648;
  modelica_real tmp1649;
  tmp1640 = (data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */);
  tmp1641 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1642 = (tmp1640 * tmp1640) + (tmp1641 * tmp1641);
  tmp1643 = 1.5;
  if(tmp1642 < 0.0 && tmp1643 != 0.0)
  {
    tmp1645 = modf(tmp1643, &tmp1646);
    
    if(tmp1645 > 0.5)
    {
      tmp1645 -= 1.0;
      tmp1646 += 1.0;
    }
    else if(tmp1645 < -0.5)
    {
      tmp1645 += 1.0;
      tmp1646 -= 1.0;
    }
    
    if(fabs(tmp1645) < 1e-10)
      tmp1644 = pow(tmp1642, tmp1646);
    else
    {
      tmp1648 = modf(1.0/tmp1643, &tmp1647);
      if(tmp1648 > 0.5)
      {
        tmp1648 -= 1.0;
        tmp1647 += 1.0;
      }
      else if(tmp1648 < -0.5)
      {
        tmp1648 += 1.0;
        tmp1647 -= 1.0;
      }
      if(fabs(tmp1648) < 1e-10 && ((unsigned long)tmp1647 & 1))
      {
        tmp1644 = -pow(-tmp1642, tmp1645)*pow(tmp1642, tmp1646);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1642, tmp1643);
      }
    }
  }
  else
  {
    tmp1644 = pow(tmp1642, tmp1643);
  }
  if(isnan(tmp1644) || isinf(tmp1644))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1642, tmp1643);
  }tmp1649 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1644,"(r_init[336] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1649 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[336] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1649);
    }
  }
  (data->simulationInfo->realParameter[840] /* omega_c[336] PARAM */) = sqrt(tmp1649);
  TRACE_POP
}

/*
equation index: 13331
type: SIMPLE_ASSIGN
r_init[335] = r_min + 335.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13331(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13331};
  (data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (335.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13332
type: SIMPLE_ASSIGN
omega_c[335] = sqrt(G * Md / (r_init[335] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13332};
  modelica_real tmp1650;
  modelica_real tmp1651;
  modelica_real tmp1652;
  modelica_real tmp1653;
  modelica_real tmp1654;
  modelica_real tmp1655;
  modelica_real tmp1656;
  modelica_real tmp1657;
  modelica_real tmp1658;
  modelica_real tmp1659;
  tmp1650 = (data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */);
  tmp1651 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1652 = (tmp1650 * tmp1650) + (tmp1651 * tmp1651);
  tmp1653 = 1.5;
  if(tmp1652 < 0.0 && tmp1653 != 0.0)
  {
    tmp1655 = modf(tmp1653, &tmp1656);
    
    if(tmp1655 > 0.5)
    {
      tmp1655 -= 1.0;
      tmp1656 += 1.0;
    }
    else if(tmp1655 < -0.5)
    {
      tmp1655 += 1.0;
      tmp1656 -= 1.0;
    }
    
    if(fabs(tmp1655) < 1e-10)
      tmp1654 = pow(tmp1652, tmp1656);
    else
    {
      tmp1658 = modf(1.0/tmp1653, &tmp1657);
      if(tmp1658 > 0.5)
      {
        tmp1658 -= 1.0;
        tmp1657 += 1.0;
      }
      else if(tmp1658 < -0.5)
      {
        tmp1658 += 1.0;
        tmp1657 -= 1.0;
      }
      if(fabs(tmp1658) < 1e-10 && ((unsigned long)tmp1657 & 1))
      {
        tmp1654 = -pow(-tmp1652, tmp1655)*pow(tmp1652, tmp1656);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1652, tmp1653);
      }
    }
  }
  else
  {
    tmp1654 = pow(tmp1652, tmp1653);
  }
  if(isnan(tmp1654) || isinf(tmp1654))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1652, tmp1653);
  }tmp1659 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1654,"(r_init[335] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1659 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[335] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1659);
    }
  }
  (data->simulationInfo->realParameter[839] /* omega_c[335] PARAM */) = sqrt(tmp1659);
  TRACE_POP
}

/*
equation index: 13333
type: SIMPLE_ASSIGN
r_init[334] = r_min + 334.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13333};
  (data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (334.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13334
type: SIMPLE_ASSIGN
omega_c[334] = sqrt(G * Md / (r_init[334] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13334(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13334};
  modelica_real tmp1660;
  modelica_real tmp1661;
  modelica_real tmp1662;
  modelica_real tmp1663;
  modelica_real tmp1664;
  modelica_real tmp1665;
  modelica_real tmp1666;
  modelica_real tmp1667;
  modelica_real tmp1668;
  modelica_real tmp1669;
  tmp1660 = (data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */);
  tmp1661 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1662 = (tmp1660 * tmp1660) + (tmp1661 * tmp1661);
  tmp1663 = 1.5;
  if(tmp1662 < 0.0 && tmp1663 != 0.0)
  {
    tmp1665 = modf(tmp1663, &tmp1666);
    
    if(tmp1665 > 0.5)
    {
      tmp1665 -= 1.0;
      tmp1666 += 1.0;
    }
    else if(tmp1665 < -0.5)
    {
      tmp1665 += 1.0;
      tmp1666 -= 1.0;
    }
    
    if(fabs(tmp1665) < 1e-10)
      tmp1664 = pow(tmp1662, tmp1666);
    else
    {
      tmp1668 = modf(1.0/tmp1663, &tmp1667);
      if(tmp1668 > 0.5)
      {
        tmp1668 -= 1.0;
        tmp1667 += 1.0;
      }
      else if(tmp1668 < -0.5)
      {
        tmp1668 += 1.0;
        tmp1667 -= 1.0;
      }
      if(fabs(tmp1668) < 1e-10 && ((unsigned long)tmp1667 & 1))
      {
        tmp1664 = -pow(-tmp1662, tmp1665)*pow(tmp1662, tmp1666);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1662, tmp1663);
      }
    }
  }
  else
  {
    tmp1664 = pow(tmp1662, tmp1663);
  }
  if(isnan(tmp1664) || isinf(tmp1664))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1662, tmp1663);
  }tmp1669 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1664,"(r_init[334] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1669 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[334] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1669);
    }
  }
  (data->simulationInfo->realParameter[838] /* omega_c[334] PARAM */) = sqrt(tmp1669);
  TRACE_POP
}

/*
equation index: 13335
type: SIMPLE_ASSIGN
r_init[333] = r_min + 333.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13335};
  (data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (333.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13336
type: SIMPLE_ASSIGN
omega_c[333] = sqrt(G * Md / (r_init[333] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13336(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13336};
  modelica_real tmp1670;
  modelica_real tmp1671;
  modelica_real tmp1672;
  modelica_real tmp1673;
  modelica_real tmp1674;
  modelica_real tmp1675;
  modelica_real tmp1676;
  modelica_real tmp1677;
  modelica_real tmp1678;
  modelica_real tmp1679;
  tmp1670 = (data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */);
  tmp1671 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1672 = (tmp1670 * tmp1670) + (tmp1671 * tmp1671);
  tmp1673 = 1.5;
  if(tmp1672 < 0.0 && tmp1673 != 0.0)
  {
    tmp1675 = modf(tmp1673, &tmp1676);
    
    if(tmp1675 > 0.5)
    {
      tmp1675 -= 1.0;
      tmp1676 += 1.0;
    }
    else if(tmp1675 < -0.5)
    {
      tmp1675 += 1.0;
      tmp1676 -= 1.0;
    }
    
    if(fabs(tmp1675) < 1e-10)
      tmp1674 = pow(tmp1672, tmp1676);
    else
    {
      tmp1678 = modf(1.0/tmp1673, &tmp1677);
      if(tmp1678 > 0.5)
      {
        tmp1678 -= 1.0;
        tmp1677 += 1.0;
      }
      else if(tmp1678 < -0.5)
      {
        tmp1678 += 1.0;
        tmp1677 -= 1.0;
      }
      if(fabs(tmp1678) < 1e-10 && ((unsigned long)tmp1677 & 1))
      {
        tmp1674 = -pow(-tmp1672, tmp1675)*pow(tmp1672, tmp1676);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1672, tmp1673);
      }
    }
  }
  else
  {
    tmp1674 = pow(tmp1672, tmp1673);
  }
  if(isnan(tmp1674) || isinf(tmp1674))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1672, tmp1673);
  }tmp1679 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1674,"(r_init[333] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1679 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[333] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1679);
    }
  }
  (data->simulationInfo->realParameter[837] /* omega_c[333] PARAM */) = sqrt(tmp1679);
  TRACE_POP
}

/*
equation index: 13337
type: SIMPLE_ASSIGN
r_init[332] = r_min + 332.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13337};
  (data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (332.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13338
type: SIMPLE_ASSIGN
omega_c[332] = sqrt(G * Md / (r_init[332] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13338(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13338};
  modelica_real tmp1680;
  modelica_real tmp1681;
  modelica_real tmp1682;
  modelica_real tmp1683;
  modelica_real tmp1684;
  modelica_real tmp1685;
  modelica_real tmp1686;
  modelica_real tmp1687;
  modelica_real tmp1688;
  modelica_real tmp1689;
  tmp1680 = (data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */);
  tmp1681 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1682 = (tmp1680 * tmp1680) + (tmp1681 * tmp1681);
  tmp1683 = 1.5;
  if(tmp1682 < 0.0 && tmp1683 != 0.0)
  {
    tmp1685 = modf(tmp1683, &tmp1686);
    
    if(tmp1685 > 0.5)
    {
      tmp1685 -= 1.0;
      tmp1686 += 1.0;
    }
    else if(tmp1685 < -0.5)
    {
      tmp1685 += 1.0;
      tmp1686 -= 1.0;
    }
    
    if(fabs(tmp1685) < 1e-10)
      tmp1684 = pow(tmp1682, tmp1686);
    else
    {
      tmp1688 = modf(1.0/tmp1683, &tmp1687);
      if(tmp1688 > 0.5)
      {
        tmp1688 -= 1.0;
        tmp1687 += 1.0;
      }
      else if(tmp1688 < -0.5)
      {
        tmp1688 += 1.0;
        tmp1687 -= 1.0;
      }
      if(fabs(tmp1688) < 1e-10 && ((unsigned long)tmp1687 & 1))
      {
        tmp1684 = -pow(-tmp1682, tmp1685)*pow(tmp1682, tmp1686);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1682, tmp1683);
      }
    }
  }
  else
  {
    tmp1684 = pow(tmp1682, tmp1683);
  }
  if(isnan(tmp1684) || isinf(tmp1684))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1682, tmp1683);
  }tmp1689 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1684,"(r_init[332] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1689 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[332] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1689);
    }
  }
  (data->simulationInfo->realParameter[836] /* omega_c[332] PARAM */) = sqrt(tmp1689);
  TRACE_POP
}

/*
equation index: 13339
type: SIMPLE_ASSIGN
r_init[331] = r_min + 331.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13339};
  (data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (331.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13340
type: SIMPLE_ASSIGN
omega_c[331] = sqrt(G * Md / (r_init[331] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13340};
  modelica_real tmp1690;
  modelica_real tmp1691;
  modelica_real tmp1692;
  modelica_real tmp1693;
  modelica_real tmp1694;
  modelica_real tmp1695;
  modelica_real tmp1696;
  modelica_real tmp1697;
  modelica_real tmp1698;
  modelica_real tmp1699;
  tmp1690 = (data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */);
  tmp1691 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1692 = (tmp1690 * tmp1690) + (tmp1691 * tmp1691);
  tmp1693 = 1.5;
  if(tmp1692 < 0.0 && tmp1693 != 0.0)
  {
    tmp1695 = modf(tmp1693, &tmp1696);
    
    if(tmp1695 > 0.5)
    {
      tmp1695 -= 1.0;
      tmp1696 += 1.0;
    }
    else if(tmp1695 < -0.5)
    {
      tmp1695 += 1.0;
      tmp1696 -= 1.0;
    }
    
    if(fabs(tmp1695) < 1e-10)
      tmp1694 = pow(tmp1692, tmp1696);
    else
    {
      tmp1698 = modf(1.0/tmp1693, &tmp1697);
      if(tmp1698 > 0.5)
      {
        tmp1698 -= 1.0;
        tmp1697 += 1.0;
      }
      else if(tmp1698 < -0.5)
      {
        tmp1698 += 1.0;
        tmp1697 -= 1.0;
      }
      if(fabs(tmp1698) < 1e-10 && ((unsigned long)tmp1697 & 1))
      {
        tmp1694 = -pow(-tmp1692, tmp1695)*pow(tmp1692, tmp1696);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1692, tmp1693);
      }
    }
  }
  else
  {
    tmp1694 = pow(tmp1692, tmp1693);
  }
  if(isnan(tmp1694) || isinf(tmp1694))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1692, tmp1693);
  }tmp1699 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1694,"(r_init[331] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1699 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[331] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1699);
    }
  }
  (data->simulationInfo->realParameter[835] /* omega_c[331] PARAM */) = sqrt(tmp1699);
  TRACE_POP
}

/*
equation index: 13341
type: SIMPLE_ASSIGN
r_init[330] = r_min + 330.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13341};
  (data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (330.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13342
type: SIMPLE_ASSIGN
omega_c[330] = sqrt(G * Md / (r_init[330] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13342(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13342};
  modelica_real tmp1700;
  modelica_real tmp1701;
  modelica_real tmp1702;
  modelica_real tmp1703;
  modelica_real tmp1704;
  modelica_real tmp1705;
  modelica_real tmp1706;
  modelica_real tmp1707;
  modelica_real tmp1708;
  modelica_real tmp1709;
  tmp1700 = (data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */);
  tmp1701 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1702 = (tmp1700 * tmp1700) + (tmp1701 * tmp1701);
  tmp1703 = 1.5;
  if(tmp1702 < 0.0 && tmp1703 != 0.0)
  {
    tmp1705 = modf(tmp1703, &tmp1706);
    
    if(tmp1705 > 0.5)
    {
      tmp1705 -= 1.0;
      tmp1706 += 1.0;
    }
    else if(tmp1705 < -0.5)
    {
      tmp1705 += 1.0;
      tmp1706 -= 1.0;
    }
    
    if(fabs(tmp1705) < 1e-10)
      tmp1704 = pow(tmp1702, tmp1706);
    else
    {
      tmp1708 = modf(1.0/tmp1703, &tmp1707);
      if(tmp1708 > 0.5)
      {
        tmp1708 -= 1.0;
        tmp1707 += 1.0;
      }
      else if(tmp1708 < -0.5)
      {
        tmp1708 += 1.0;
        tmp1707 -= 1.0;
      }
      if(fabs(tmp1708) < 1e-10 && ((unsigned long)tmp1707 & 1))
      {
        tmp1704 = -pow(-tmp1702, tmp1705)*pow(tmp1702, tmp1706);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1702, tmp1703);
      }
    }
  }
  else
  {
    tmp1704 = pow(tmp1702, tmp1703);
  }
  if(isnan(tmp1704) || isinf(tmp1704))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1702, tmp1703);
  }tmp1709 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1704,"(r_init[330] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1709 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[330] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1709);
    }
  }
  (data->simulationInfo->realParameter[834] /* omega_c[330] PARAM */) = sqrt(tmp1709);
  TRACE_POP
}

/*
equation index: 13343
type: SIMPLE_ASSIGN
r_init[329] = r_min + 329.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13343};
  (data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (329.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13344
type: SIMPLE_ASSIGN
omega_c[329] = sqrt(G * Md / (r_init[329] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13344(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13344};
  modelica_real tmp1710;
  modelica_real tmp1711;
  modelica_real tmp1712;
  modelica_real tmp1713;
  modelica_real tmp1714;
  modelica_real tmp1715;
  modelica_real tmp1716;
  modelica_real tmp1717;
  modelica_real tmp1718;
  modelica_real tmp1719;
  tmp1710 = (data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */);
  tmp1711 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1712 = (tmp1710 * tmp1710) + (tmp1711 * tmp1711);
  tmp1713 = 1.5;
  if(tmp1712 < 0.0 && tmp1713 != 0.0)
  {
    tmp1715 = modf(tmp1713, &tmp1716);
    
    if(tmp1715 > 0.5)
    {
      tmp1715 -= 1.0;
      tmp1716 += 1.0;
    }
    else if(tmp1715 < -0.5)
    {
      tmp1715 += 1.0;
      tmp1716 -= 1.0;
    }
    
    if(fabs(tmp1715) < 1e-10)
      tmp1714 = pow(tmp1712, tmp1716);
    else
    {
      tmp1718 = modf(1.0/tmp1713, &tmp1717);
      if(tmp1718 > 0.5)
      {
        tmp1718 -= 1.0;
        tmp1717 += 1.0;
      }
      else if(tmp1718 < -0.5)
      {
        tmp1718 += 1.0;
        tmp1717 -= 1.0;
      }
      if(fabs(tmp1718) < 1e-10 && ((unsigned long)tmp1717 & 1))
      {
        tmp1714 = -pow(-tmp1712, tmp1715)*pow(tmp1712, tmp1716);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1712, tmp1713);
      }
    }
  }
  else
  {
    tmp1714 = pow(tmp1712, tmp1713);
  }
  if(isnan(tmp1714) || isinf(tmp1714))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1712, tmp1713);
  }tmp1719 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1714,"(r_init[329] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1719 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[329] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1719);
    }
  }
  (data->simulationInfo->realParameter[833] /* omega_c[329] PARAM */) = sqrt(tmp1719);
  TRACE_POP
}

/*
equation index: 13345
type: SIMPLE_ASSIGN
r_init[328] = r_min + 328.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13345};
  (data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (328.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13346
type: SIMPLE_ASSIGN
omega_c[328] = sqrt(G * Md / (r_init[328] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13346(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13346};
  modelica_real tmp1720;
  modelica_real tmp1721;
  modelica_real tmp1722;
  modelica_real tmp1723;
  modelica_real tmp1724;
  modelica_real tmp1725;
  modelica_real tmp1726;
  modelica_real tmp1727;
  modelica_real tmp1728;
  modelica_real tmp1729;
  tmp1720 = (data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */);
  tmp1721 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1722 = (tmp1720 * tmp1720) + (tmp1721 * tmp1721);
  tmp1723 = 1.5;
  if(tmp1722 < 0.0 && tmp1723 != 0.0)
  {
    tmp1725 = modf(tmp1723, &tmp1726);
    
    if(tmp1725 > 0.5)
    {
      tmp1725 -= 1.0;
      tmp1726 += 1.0;
    }
    else if(tmp1725 < -0.5)
    {
      tmp1725 += 1.0;
      tmp1726 -= 1.0;
    }
    
    if(fabs(tmp1725) < 1e-10)
      tmp1724 = pow(tmp1722, tmp1726);
    else
    {
      tmp1728 = modf(1.0/tmp1723, &tmp1727);
      if(tmp1728 > 0.5)
      {
        tmp1728 -= 1.0;
        tmp1727 += 1.0;
      }
      else if(tmp1728 < -0.5)
      {
        tmp1728 += 1.0;
        tmp1727 -= 1.0;
      }
      if(fabs(tmp1728) < 1e-10 && ((unsigned long)tmp1727 & 1))
      {
        tmp1724 = -pow(-tmp1722, tmp1725)*pow(tmp1722, tmp1726);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1722, tmp1723);
      }
    }
  }
  else
  {
    tmp1724 = pow(tmp1722, tmp1723);
  }
  if(isnan(tmp1724) || isinf(tmp1724))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1722, tmp1723);
  }tmp1729 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1724,"(r_init[328] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1729 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[328] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1729);
    }
  }
  (data->simulationInfo->realParameter[832] /* omega_c[328] PARAM */) = sqrt(tmp1729);
  TRACE_POP
}

/*
equation index: 13347
type: SIMPLE_ASSIGN
r_init[327] = r_min + 327.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13347(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13347};
  (data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (327.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13348
type: SIMPLE_ASSIGN
omega_c[327] = sqrt(G * Md / (r_init[327] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13348};
  modelica_real tmp1730;
  modelica_real tmp1731;
  modelica_real tmp1732;
  modelica_real tmp1733;
  modelica_real tmp1734;
  modelica_real tmp1735;
  modelica_real tmp1736;
  modelica_real tmp1737;
  modelica_real tmp1738;
  modelica_real tmp1739;
  tmp1730 = (data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */);
  tmp1731 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1732 = (tmp1730 * tmp1730) + (tmp1731 * tmp1731);
  tmp1733 = 1.5;
  if(tmp1732 < 0.0 && tmp1733 != 0.0)
  {
    tmp1735 = modf(tmp1733, &tmp1736);
    
    if(tmp1735 > 0.5)
    {
      tmp1735 -= 1.0;
      tmp1736 += 1.0;
    }
    else if(tmp1735 < -0.5)
    {
      tmp1735 += 1.0;
      tmp1736 -= 1.0;
    }
    
    if(fabs(tmp1735) < 1e-10)
      tmp1734 = pow(tmp1732, tmp1736);
    else
    {
      tmp1738 = modf(1.0/tmp1733, &tmp1737);
      if(tmp1738 > 0.5)
      {
        tmp1738 -= 1.0;
        tmp1737 += 1.0;
      }
      else if(tmp1738 < -0.5)
      {
        tmp1738 += 1.0;
        tmp1737 -= 1.0;
      }
      if(fabs(tmp1738) < 1e-10 && ((unsigned long)tmp1737 & 1))
      {
        tmp1734 = -pow(-tmp1732, tmp1735)*pow(tmp1732, tmp1736);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1732, tmp1733);
      }
    }
  }
  else
  {
    tmp1734 = pow(tmp1732, tmp1733);
  }
  if(isnan(tmp1734) || isinf(tmp1734))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1732, tmp1733);
  }tmp1739 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1734,"(r_init[327] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1739 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[327] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1739);
    }
  }
  (data->simulationInfo->realParameter[831] /* omega_c[327] PARAM */) = sqrt(tmp1739);
  TRACE_POP
}

/*
equation index: 13349
type: SIMPLE_ASSIGN
r_init[326] = r_min + 326.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13349};
  (data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (326.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13350
type: SIMPLE_ASSIGN
omega_c[326] = sqrt(G * Md / (r_init[326] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13350(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13350};
  modelica_real tmp1740;
  modelica_real tmp1741;
  modelica_real tmp1742;
  modelica_real tmp1743;
  modelica_real tmp1744;
  modelica_real tmp1745;
  modelica_real tmp1746;
  modelica_real tmp1747;
  modelica_real tmp1748;
  modelica_real tmp1749;
  tmp1740 = (data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */);
  tmp1741 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1742 = (tmp1740 * tmp1740) + (tmp1741 * tmp1741);
  tmp1743 = 1.5;
  if(tmp1742 < 0.0 && tmp1743 != 0.0)
  {
    tmp1745 = modf(tmp1743, &tmp1746);
    
    if(tmp1745 > 0.5)
    {
      tmp1745 -= 1.0;
      tmp1746 += 1.0;
    }
    else if(tmp1745 < -0.5)
    {
      tmp1745 += 1.0;
      tmp1746 -= 1.0;
    }
    
    if(fabs(tmp1745) < 1e-10)
      tmp1744 = pow(tmp1742, tmp1746);
    else
    {
      tmp1748 = modf(1.0/tmp1743, &tmp1747);
      if(tmp1748 > 0.5)
      {
        tmp1748 -= 1.0;
        tmp1747 += 1.0;
      }
      else if(tmp1748 < -0.5)
      {
        tmp1748 += 1.0;
        tmp1747 -= 1.0;
      }
      if(fabs(tmp1748) < 1e-10 && ((unsigned long)tmp1747 & 1))
      {
        tmp1744 = -pow(-tmp1742, tmp1745)*pow(tmp1742, tmp1746);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1742, tmp1743);
      }
    }
  }
  else
  {
    tmp1744 = pow(tmp1742, tmp1743);
  }
  if(isnan(tmp1744) || isinf(tmp1744))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1742, tmp1743);
  }tmp1749 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1744,"(r_init[326] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1749 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[326] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1749);
    }
  }
  (data->simulationInfo->realParameter[830] /* omega_c[326] PARAM */) = sqrt(tmp1749);
  TRACE_POP
}

/*
equation index: 13351
type: SIMPLE_ASSIGN
r_init[325] = r_min + 325.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13351};
  (data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (325.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13352
type: SIMPLE_ASSIGN
omega_c[325] = sqrt(G * Md / (r_init[325] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13352(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13352};
  modelica_real tmp1750;
  modelica_real tmp1751;
  modelica_real tmp1752;
  modelica_real tmp1753;
  modelica_real tmp1754;
  modelica_real tmp1755;
  modelica_real tmp1756;
  modelica_real tmp1757;
  modelica_real tmp1758;
  modelica_real tmp1759;
  tmp1750 = (data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */);
  tmp1751 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1752 = (tmp1750 * tmp1750) + (tmp1751 * tmp1751);
  tmp1753 = 1.5;
  if(tmp1752 < 0.0 && tmp1753 != 0.0)
  {
    tmp1755 = modf(tmp1753, &tmp1756);
    
    if(tmp1755 > 0.5)
    {
      tmp1755 -= 1.0;
      tmp1756 += 1.0;
    }
    else if(tmp1755 < -0.5)
    {
      tmp1755 += 1.0;
      tmp1756 -= 1.0;
    }
    
    if(fabs(tmp1755) < 1e-10)
      tmp1754 = pow(tmp1752, tmp1756);
    else
    {
      tmp1758 = modf(1.0/tmp1753, &tmp1757);
      if(tmp1758 > 0.5)
      {
        tmp1758 -= 1.0;
        tmp1757 += 1.0;
      }
      else if(tmp1758 < -0.5)
      {
        tmp1758 += 1.0;
        tmp1757 -= 1.0;
      }
      if(fabs(tmp1758) < 1e-10 && ((unsigned long)tmp1757 & 1))
      {
        tmp1754 = -pow(-tmp1752, tmp1755)*pow(tmp1752, tmp1756);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1752, tmp1753);
      }
    }
  }
  else
  {
    tmp1754 = pow(tmp1752, tmp1753);
  }
  if(isnan(tmp1754) || isinf(tmp1754))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1752, tmp1753);
  }tmp1759 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1754,"(r_init[325] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1759 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[325] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1759);
    }
  }
  (data->simulationInfo->realParameter[829] /* omega_c[325] PARAM */) = sqrt(tmp1759);
  TRACE_POP
}

/*
equation index: 13353
type: SIMPLE_ASSIGN
r_init[324] = r_min + 324.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13353};
  (data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (324.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13354
type: SIMPLE_ASSIGN
omega_c[324] = sqrt(G * Md / (r_init[324] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13354(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13354};
  modelica_real tmp1760;
  modelica_real tmp1761;
  modelica_real tmp1762;
  modelica_real tmp1763;
  modelica_real tmp1764;
  modelica_real tmp1765;
  modelica_real tmp1766;
  modelica_real tmp1767;
  modelica_real tmp1768;
  modelica_real tmp1769;
  tmp1760 = (data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */);
  tmp1761 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1762 = (tmp1760 * tmp1760) + (tmp1761 * tmp1761);
  tmp1763 = 1.5;
  if(tmp1762 < 0.0 && tmp1763 != 0.0)
  {
    tmp1765 = modf(tmp1763, &tmp1766);
    
    if(tmp1765 > 0.5)
    {
      tmp1765 -= 1.0;
      tmp1766 += 1.0;
    }
    else if(tmp1765 < -0.5)
    {
      tmp1765 += 1.0;
      tmp1766 -= 1.0;
    }
    
    if(fabs(tmp1765) < 1e-10)
      tmp1764 = pow(tmp1762, tmp1766);
    else
    {
      tmp1768 = modf(1.0/tmp1763, &tmp1767);
      if(tmp1768 > 0.5)
      {
        tmp1768 -= 1.0;
        tmp1767 += 1.0;
      }
      else if(tmp1768 < -0.5)
      {
        tmp1768 += 1.0;
        tmp1767 -= 1.0;
      }
      if(fabs(tmp1768) < 1e-10 && ((unsigned long)tmp1767 & 1))
      {
        tmp1764 = -pow(-tmp1762, tmp1765)*pow(tmp1762, tmp1766);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1762, tmp1763);
      }
    }
  }
  else
  {
    tmp1764 = pow(tmp1762, tmp1763);
  }
  if(isnan(tmp1764) || isinf(tmp1764))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1762, tmp1763);
  }tmp1769 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1764,"(r_init[324] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1769 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[324] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1769);
    }
  }
  (data->simulationInfo->realParameter[828] /* omega_c[324] PARAM */) = sqrt(tmp1769);
  TRACE_POP
}

/*
equation index: 13355
type: SIMPLE_ASSIGN
r_init[323] = r_min + 323.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13355};
  (data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (323.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13356
type: SIMPLE_ASSIGN
omega_c[323] = sqrt(G * Md / (r_init[323] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13356};
  modelica_real tmp1770;
  modelica_real tmp1771;
  modelica_real tmp1772;
  modelica_real tmp1773;
  modelica_real tmp1774;
  modelica_real tmp1775;
  modelica_real tmp1776;
  modelica_real tmp1777;
  modelica_real tmp1778;
  modelica_real tmp1779;
  tmp1770 = (data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */);
  tmp1771 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1772 = (tmp1770 * tmp1770) + (tmp1771 * tmp1771);
  tmp1773 = 1.5;
  if(tmp1772 < 0.0 && tmp1773 != 0.0)
  {
    tmp1775 = modf(tmp1773, &tmp1776);
    
    if(tmp1775 > 0.5)
    {
      tmp1775 -= 1.0;
      tmp1776 += 1.0;
    }
    else if(tmp1775 < -0.5)
    {
      tmp1775 += 1.0;
      tmp1776 -= 1.0;
    }
    
    if(fabs(tmp1775) < 1e-10)
      tmp1774 = pow(tmp1772, tmp1776);
    else
    {
      tmp1778 = modf(1.0/tmp1773, &tmp1777);
      if(tmp1778 > 0.5)
      {
        tmp1778 -= 1.0;
        tmp1777 += 1.0;
      }
      else if(tmp1778 < -0.5)
      {
        tmp1778 += 1.0;
        tmp1777 -= 1.0;
      }
      if(fabs(tmp1778) < 1e-10 && ((unsigned long)tmp1777 & 1))
      {
        tmp1774 = -pow(-tmp1772, tmp1775)*pow(tmp1772, tmp1776);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1772, tmp1773);
      }
    }
  }
  else
  {
    tmp1774 = pow(tmp1772, tmp1773);
  }
  if(isnan(tmp1774) || isinf(tmp1774))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1772, tmp1773);
  }tmp1779 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1774,"(r_init[323] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1779 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[323] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1779);
    }
  }
  (data->simulationInfo->realParameter[827] /* omega_c[323] PARAM */) = sqrt(tmp1779);
  TRACE_POP
}

/*
equation index: 13357
type: SIMPLE_ASSIGN
r_init[322] = r_min + 322.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13357};
  (data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (322.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13358
type: SIMPLE_ASSIGN
omega_c[322] = sqrt(G * Md / (r_init[322] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13358(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13358};
  modelica_real tmp1780;
  modelica_real tmp1781;
  modelica_real tmp1782;
  modelica_real tmp1783;
  modelica_real tmp1784;
  modelica_real tmp1785;
  modelica_real tmp1786;
  modelica_real tmp1787;
  modelica_real tmp1788;
  modelica_real tmp1789;
  tmp1780 = (data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */);
  tmp1781 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1782 = (tmp1780 * tmp1780) + (tmp1781 * tmp1781);
  tmp1783 = 1.5;
  if(tmp1782 < 0.0 && tmp1783 != 0.0)
  {
    tmp1785 = modf(tmp1783, &tmp1786);
    
    if(tmp1785 > 0.5)
    {
      tmp1785 -= 1.0;
      tmp1786 += 1.0;
    }
    else if(tmp1785 < -0.5)
    {
      tmp1785 += 1.0;
      tmp1786 -= 1.0;
    }
    
    if(fabs(tmp1785) < 1e-10)
      tmp1784 = pow(tmp1782, tmp1786);
    else
    {
      tmp1788 = modf(1.0/tmp1783, &tmp1787);
      if(tmp1788 > 0.5)
      {
        tmp1788 -= 1.0;
        tmp1787 += 1.0;
      }
      else if(tmp1788 < -0.5)
      {
        tmp1788 += 1.0;
        tmp1787 -= 1.0;
      }
      if(fabs(tmp1788) < 1e-10 && ((unsigned long)tmp1787 & 1))
      {
        tmp1784 = -pow(-tmp1782, tmp1785)*pow(tmp1782, tmp1786);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1782, tmp1783);
      }
    }
  }
  else
  {
    tmp1784 = pow(tmp1782, tmp1783);
  }
  if(isnan(tmp1784) || isinf(tmp1784))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1782, tmp1783);
  }tmp1789 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1784,"(r_init[322] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1789 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[322] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1789);
    }
  }
  (data->simulationInfo->realParameter[826] /* omega_c[322] PARAM */) = sqrt(tmp1789);
  TRACE_POP
}

/*
equation index: 13359
type: SIMPLE_ASSIGN
r_init[321] = r_min + 321.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13359};
  (data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (321.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13360
type: SIMPLE_ASSIGN
omega_c[321] = sqrt(G * Md / (r_init[321] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13360(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13360};
  modelica_real tmp1790;
  modelica_real tmp1791;
  modelica_real tmp1792;
  modelica_real tmp1793;
  modelica_real tmp1794;
  modelica_real tmp1795;
  modelica_real tmp1796;
  modelica_real tmp1797;
  modelica_real tmp1798;
  modelica_real tmp1799;
  tmp1790 = (data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */);
  tmp1791 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1792 = (tmp1790 * tmp1790) + (tmp1791 * tmp1791);
  tmp1793 = 1.5;
  if(tmp1792 < 0.0 && tmp1793 != 0.0)
  {
    tmp1795 = modf(tmp1793, &tmp1796);
    
    if(tmp1795 > 0.5)
    {
      tmp1795 -= 1.0;
      tmp1796 += 1.0;
    }
    else if(tmp1795 < -0.5)
    {
      tmp1795 += 1.0;
      tmp1796 -= 1.0;
    }
    
    if(fabs(tmp1795) < 1e-10)
      tmp1794 = pow(tmp1792, tmp1796);
    else
    {
      tmp1798 = modf(1.0/tmp1793, &tmp1797);
      if(tmp1798 > 0.5)
      {
        tmp1798 -= 1.0;
        tmp1797 += 1.0;
      }
      else if(tmp1798 < -0.5)
      {
        tmp1798 += 1.0;
        tmp1797 -= 1.0;
      }
      if(fabs(tmp1798) < 1e-10 && ((unsigned long)tmp1797 & 1))
      {
        tmp1794 = -pow(-tmp1792, tmp1795)*pow(tmp1792, tmp1796);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1792, tmp1793);
      }
    }
  }
  else
  {
    tmp1794 = pow(tmp1792, tmp1793);
  }
  if(isnan(tmp1794) || isinf(tmp1794))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1792, tmp1793);
  }tmp1799 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1794,"(r_init[321] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1799 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[321] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1799);
    }
  }
  (data->simulationInfo->realParameter[825] /* omega_c[321] PARAM */) = sqrt(tmp1799);
  TRACE_POP
}

/*
equation index: 13361
type: SIMPLE_ASSIGN
r_init[320] = r_min + 320.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13361};
  (data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (320.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13362
type: SIMPLE_ASSIGN
omega_c[320] = sqrt(G * Md / (r_init[320] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13362(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13362};
  modelica_real tmp1800;
  modelica_real tmp1801;
  modelica_real tmp1802;
  modelica_real tmp1803;
  modelica_real tmp1804;
  modelica_real tmp1805;
  modelica_real tmp1806;
  modelica_real tmp1807;
  modelica_real tmp1808;
  modelica_real tmp1809;
  tmp1800 = (data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */);
  tmp1801 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1802 = (tmp1800 * tmp1800) + (tmp1801 * tmp1801);
  tmp1803 = 1.5;
  if(tmp1802 < 0.0 && tmp1803 != 0.0)
  {
    tmp1805 = modf(tmp1803, &tmp1806);
    
    if(tmp1805 > 0.5)
    {
      tmp1805 -= 1.0;
      tmp1806 += 1.0;
    }
    else if(tmp1805 < -0.5)
    {
      tmp1805 += 1.0;
      tmp1806 -= 1.0;
    }
    
    if(fabs(tmp1805) < 1e-10)
      tmp1804 = pow(tmp1802, tmp1806);
    else
    {
      tmp1808 = modf(1.0/tmp1803, &tmp1807);
      if(tmp1808 > 0.5)
      {
        tmp1808 -= 1.0;
        tmp1807 += 1.0;
      }
      else if(tmp1808 < -0.5)
      {
        tmp1808 += 1.0;
        tmp1807 -= 1.0;
      }
      if(fabs(tmp1808) < 1e-10 && ((unsigned long)tmp1807 & 1))
      {
        tmp1804 = -pow(-tmp1802, tmp1805)*pow(tmp1802, tmp1806);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1802, tmp1803);
      }
    }
  }
  else
  {
    tmp1804 = pow(tmp1802, tmp1803);
  }
  if(isnan(tmp1804) || isinf(tmp1804))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1802, tmp1803);
  }tmp1809 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1804,"(r_init[320] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1809 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[320] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1809);
    }
  }
  (data->simulationInfo->realParameter[824] /* omega_c[320] PARAM */) = sqrt(tmp1809);
  TRACE_POP
}

/*
equation index: 13363
type: SIMPLE_ASSIGN
r_init[319] = r_min + 319.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13363};
  (data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (319.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13364
type: SIMPLE_ASSIGN
omega_c[319] = sqrt(G * Md / (r_init[319] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13364};
  modelica_real tmp1810;
  modelica_real tmp1811;
  modelica_real tmp1812;
  modelica_real tmp1813;
  modelica_real tmp1814;
  modelica_real tmp1815;
  modelica_real tmp1816;
  modelica_real tmp1817;
  modelica_real tmp1818;
  modelica_real tmp1819;
  tmp1810 = (data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */);
  tmp1811 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1812 = (tmp1810 * tmp1810) + (tmp1811 * tmp1811);
  tmp1813 = 1.5;
  if(tmp1812 < 0.0 && tmp1813 != 0.0)
  {
    tmp1815 = modf(tmp1813, &tmp1816);
    
    if(tmp1815 > 0.5)
    {
      tmp1815 -= 1.0;
      tmp1816 += 1.0;
    }
    else if(tmp1815 < -0.5)
    {
      tmp1815 += 1.0;
      tmp1816 -= 1.0;
    }
    
    if(fabs(tmp1815) < 1e-10)
      tmp1814 = pow(tmp1812, tmp1816);
    else
    {
      tmp1818 = modf(1.0/tmp1813, &tmp1817);
      if(tmp1818 > 0.5)
      {
        tmp1818 -= 1.0;
        tmp1817 += 1.0;
      }
      else if(tmp1818 < -0.5)
      {
        tmp1818 += 1.0;
        tmp1817 -= 1.0;
      }
      if(fabs(tmp1818) < 1e-10 && ((unsigned long)tmp1817 & 1))
      {
        tmp1814 = -pow(-tmp1812, tmp1815)*pow(tmp1812, tmp1816);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1812, tmp1813);
      }
    }
  }
  else
  {
    tmp1814 = pow(tmp1812, tmp1813);
  }
  if(isnan(tmp1814) || isinf(tmp1814))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1812, tmp1813);
  }tmp1819 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1814,"(r_init[319] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1819 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[319] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1819);
    }
  }
  (data->simulationInfo->realParameter[823] /* omega_c[319] PARAM */) = sqrt(tmp1819);
  TRACE_POP
}

/*
equation index: 13365
type: SIMPLE_ASSIGN
r_init[318] = r_min + 318.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13365};
  (data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (318.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13366
type: SIMPLE_ASSIGN
omega_c[318] = sqrt(G * Md / (r_init[318] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13366(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13366};
  modelica_real tmp1820;
  modelica_real tmp1821;
  modelica_real tmp1822;
  modelica_real tmp1823;
  modelica_real tmp1824;
  modelica_real tmp1825;
  modelica_real tmp1826;
  modelica_real tmp1827;
  modelica_real tmp1828;
  modelica_real tmp1829;
  tmp1820 = (data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */);
  tmp1821 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1822 = (tmp1820 * tmp1820) + (tmp1821 * tmp1821);
  tmp1823 = 1.5;
  if(tmp1822 < 0.0 && tmp1823 != 0.0)
  {
    tmp1825 = modf(tmp1823, &tmp1826);
    
    if(tmp1825 > 0.5)
    {
      tmp1825 -= 1.0;
      tmp1826 += 1.0;
    }
    else if(tmp1825 < -0.5)
    {
      tmp1825 += 1.0;
      tmp1826 -= 1.0;
    }
    
    if(fabs(tmp1825) < 1e-10)
      tmp1824 = pow(tmp1822, tmp1826);
    else
    {
      tmp1828 = modf(1.0/tmp1823, &tmp1827);
      if(tmp1828 > 0.5)
      {
        tmp1828 -= 1.0;
        tmp1827 += 1.0;
      }
      else if(tmp1828 < -0.5)
      {
        tmp1828 += 1.0;
        tmp1827 -= 1.0;
      }
      if(fabs(tmp1828) < 1e-10 && ((unsigned long)tmp1827 & 1))
      {
        tmp1824 = -pow(-tmp1822, tmp1825)*pow(tmp1822, tmp1826);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1822, tmp1823);
      }
    }
  }
  else
  {
    tmp1824 = pow(tmp1822, tmp1823);
  }
  if(isnan(tmp1824) || isinf(tmp1824))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1822, tmp1823);
  }tmp1829 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1824,"(r_init[318] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1829 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[318] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1829);
    }
  }
  (data->simulationInfo->realParameter[822] /* omega_c[318] PARAM */) = sqrt(tmp1829);
  TRACE_POP
}

/*
equation index: 13367
type: SIMPLE_ASSIGN
r_init[317] = r_min + 317.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13367};
  (data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (317.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13368
type: SIMPLE_ASSIGN
omega_c[317] = sqrt(G * Md / (r_init[317] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13368(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13368};
  modelica_real tmp1830;
  modelica_real tmp1831;
  modelica_real tmp1832;
  modelica_real tmp1833;
  modelica_real tmp1834;
  modelica_real tmp1835;
  modelica_real tmp1836;
  modelica_real tmp1837;
  modelica_real tmp1838;
  modelica_real tmp1839;
  tmp1830 = (data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */);
  tmp1831 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1832 = (tmp1830 * tmp1830) + (tmp1831 * tmp1831);
  tmp1833 = 1.5;
  if(tmp1832 < 0.0 && tmp1833 != 0.0)
  {
    tmp1835 = modf(tmp1833, &tmp1836);
    
    if(tmp1835 > 0.5)
    {
      tmp1835 -= 1.0;
      tmp1836 += 1.0;
    }
    else if(tmp1835 < -0.5)
    {
      tmp1835 += 1.0;
      tmp1836 -= 1.0;
    }
    
    if(fabs(tmp1835) < 1e-10)
      tmp1834 = pow(tmp1832, tmp1836);
    else
    {
      tmp1838 = modf(1.0/tmp1833, &tmp1837);
      if(tmp1838 > 0.5)
      {
        tmp1838 -= 1.0;
        tmp1837 += 1.0;
      }
      else if(tmp1838 < -0.5)
      {
        tmp1838 += 1.0;
        tmp1837 -= 1.0;
      }
      if(fabs(tmp1838) < 1e-10 && ((unsigned long)tmp1837 & 1))
      {
        tmp1834 = -pow(-tmp1832, tmp1835)*pow(tmp1832, tmp1836);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1832, tmp1833);
      }
    }
  }
  else
  {
    tmp1834 = pow(tmp1832, tmp1833);
  }
  if(isnan(tmp1834) || isinf(tmp1834))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1832, tmp1833);
  }tmp1839 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1834,"(r_init[317] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1839 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[317] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1839);
    }
  }
  (data->simulationInfo->realParameter[821] /* omega_c[317] PARAM */) = sqrt(tmp1839);
  TRACE_POP
}

/*
equation index: 13369
type: SIMPLE_ASSIGN
r_init[316] = r_min + 316.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13369};
  (data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (316.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13370
type: SIMPLE_ASSIGN
omega_c[316] = sqrt(G * Md / (r_init[316] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13370(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13370};
  modelica_real tmp1840;
  modelica_real tmp1841;
  modelica_real tmp1842;
  modelica_real tmp1843;
  modelica_real tmp1844;
  modelica_real tmp1845;
  modelica_real tmp1846;
  modelica_real tmp1847;
  modelica_real tmp1848;
  modelica_real tmp1849;
  tmp1840 = (data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */);
  tmp1841 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1842 = (tmp1840 * tmp1840) + (tmp1841 * tmp1841);
  tmp1843 = 1.5;
  if(tmp1842 < 0.0 && tmp1843 != 0.0)
  {
    tmp1845 = modf(tmp1843, &tmp1846);
    
    if(tmp1845 > 0.5)
    {
      tmp1845 -= 1.0;
      tmp1846 += 1.0;
    }
    else if(tmp1845 < -0.5)
    {
      tmp1845 += 1.0;
      tmp1846 -= 1.0;
    }
    
    if(fabs(tmp1845) < 1e-10)
      tmp1844 = pow(tmp1842, tmp1846);
    else
    {
      tmp1848 = modf(1.0/tmp1843, &tmp1847);
      if(tmp1848 > 0.5)
      {
        tmp1848 -= 1.0;
        tmp1847 += 1.0;
      }
      else if(tmp1848 < -0.5)
      {
        tmp1848 += 1.0;
        tmp1847 -= 1.0;
      }
      if(fabs(tmp1848) < 1e-10 && ((unsigned long)tmp1847 & 1))
      {
        tmp1844 = -pow(-tmp1842, tmp1845)*pow(tmp1842, tmp1846);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1842, tmp1843);
      }
    }
  }
  else
  {
    tmp1844 = pow(tmp1842, tmp1843);
  }
  if(isnan(tmp1844) || isinf(tmp1844))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1842, tmp1843);
  }tmp1849 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1844,"(r_init[316] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1849 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[316] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1849);
    }
  }
  (data->simulationInfo->realParameter[820] /* omega_c[316] PARAM */) = sqrt(tmp1849);
  TRACE_POP
}

/*
equation index: 13371
type: SIMPLE_ASSIGN
r_init[315] = r_min + 315.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13371};
  (data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (315.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13372
type: SIMPLE_ASSIGN
omega_c[315] = sqrt(G * Md / (r_init[315] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13372};
  modelica_real tmp1850;
  modelica_real tmp1851;
  modelica_real tmp1852;
  modelica_real tmp1853;
  modelica_real tmp1854;
  modelica_real tmp1855;
  modelica_real tmp1856;
  modelica_real tmp1857;
  modelica_real tmp1858;
  modelica_real tmp1859;
  tmp1850 = (data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */);
  tmp1851 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1852 = (tmp1850 * tmp1850) + (tmp1851 * tmp1851);
  tmp1853 = 1.5;
  if(tmp1852 < 0.0 && tmp1853 != 0.0)
  {
    tmp1855 = modf(tmp1853, &tmp1856);
    
    if(tmp1855 > 0.5)
    {
      tmp1855 -= 1.0;
      tmp1856 += 1.0;
    }
    else if(tmp1855 < -0.5)
    {
      tmp1855 += 1.0;
      tmp1856 -= 1.0;
    }
    
    if(fabs(tmp1855) < 1e-10)
      tmp1854 = pow(tmp1852, tmp1856);
    else
    {
      tmp1858 = modf(1.0/tmp1853, &tmp1857);
      if(tmp1858 > 0.5)
      {
        tmp1858 -= 1.0;
        tmp1857 += 1.0;
      }
      else if(tmp1858 < -0.5)
      {
        tmp1858 += 1.0;
        tmp1857 -= 1.0;
      }
      if(fabs(tmp1858) < 1e-10 && ((unsigned long)tmp1857 & 1))
      {
        tmp1854 = -pow(-tmp1852, tmp1855)*pow(tmp1852, tmp1856);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1852, tmp1853);
      }
    }
  }
  else
  {
    tmp1854 = pow(tmp1852, tmp1853);
  }
  if(isnan(tmp1854) || isinf(tmp1854))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1852, tmp1853);
  }tmp1859 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1854,"(r_init[315] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1859 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[315] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1859);
    }
  }
  (data->simulationInfo->realParameter[819] /* omega_c[315] PARAM */) = sqrt(tmp1859);
  TRACE_POP
}

/*
equation index: 13373
type: SIMPLE_ASSIGN
r_init[314] = r_min + 314.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13373};
  (data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (314.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13374
type: SIMPLE_ASSIGN
omega_c[314] = sqrt(G * Md / (r_init[314] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13374(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13374};
  modelica_real tmp1860;
  modelica_real tmp1861;
  modelica_real tmp1862;
  modelica_real tmp1863;
  modelica_real tmp1864;
  modelica_real tmp1865;
  modelica_real tmp1866;
  modelica_real tmp1867;
  modelica_real tmp1868;
  modelica_real tmp1869;
  tmp1860 = (data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */);
  tmp1861 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1862 = (tmp1860 * tmp1860) + (tmp1861 * tmp1861);
  tmp1863 = 1.5;
  if(tmp1862 < 0.0 && tmp1863 != 0.0)
  {
    tmp1865 = modf(tmp1863, &tmp1866);
    
    if(tmp1865 > 0.5)
    {
      tmp1865 -= 1.0;
      tmp1866 += 1.0;
    }
    else if(tmp1865 < -0.5)
    {
      tmp1865 += 1.0;
      tmp1866 -= 1.0;
    }
    
    if(fabs(tmp1865) < 1e-10)
      tmp1864 = pow(tmp1862, tmp1866);
    else
    {
      tmp1868 = modf(1.0/tmp1863, &tmp1867);
      if(tmp1868 > 0.5)
      {
        tmp1868 -= 1.0;
        tmp1867 += 1.0;
      }
      else if(tmp1868 < -0.5)
      {
        tmp1868 += 1.0;
        tmp1867 -= 1.0;
      }
      if(fabs(tmp1868) < 1e-10 && ((unsigned long)tmp1867 & 1))
      {
        tmp1864 = -pow(-tmp1862, tmp1865)*pow(tmp1862, tmp1866);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1862, tmp1863);
      }
    }
  }
  else
  {
    tmp1864 = pow(tmp1862, tmp1863);
  }
  if(isnan(tmp1864) || isinf(tmp1864))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1862, tmp1863);
  }tmp1869 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1864,"(r_init[314] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1869 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[314] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1869);
    }
  }
  (data->simulationInfo->realParameter[818] /* omega_c[314] PARAM */) = sqrt(tmp1869);
  TRACE_POP
}

/*
equation index: 13375
type: SIMPLE_ASSIGN
r_init[313] = r_min + 313.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13375};
  (data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (313.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13376
type: SIMPLE_ASSIGN
omega_c[313] = sqrt(G * Md / (r_init[313] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13376(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13376};
  modelica_real tmp1870;
  modelica_real tmp1871;
  modelica_real tmp1872;
  modelica_real tmp1873;
  modelica_real tmp1874;
  modelica_real tmp1875;
  modelica_real tmp1876;
  modelica_real tmp1877;
  modelica_real tmp1878;
  modelica_real tmp1879;
  tmp1870 = (data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */);
  tmp1871 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1872 = (tmp1870 * tmp1870) + (tmp1871 * tmp1871);
  tmp1873 = 1.5;
  if(tmp1872 < 0.0 && tmp1873 != 0.0)
  {
    tmp1875 = modf(tmp1873, &tmp1876);
    
    if(tmp1875 > 0.5)
    {
      tmp1875 -= 1.0;
      tmp1876 += 1.0;
    }
    else if(tmp1875 < -0.5)
    {
      tmp1875 += 1.0;
      tmp1876 -= 1.0;
    }
    
    if(fabs(tmp1875) < 1e-10)
      tmp1874 = pow(tmp1872, tmp1876);
    else
    {
      tmp1878 = modf(1.0/tmp1873, &tmp1877);
      if(tmp1878 > 0.5)
      {
        tmp1878 -= 1.0;
        tmp1877 += 1.0;
      }
      else if(tmp1878 < -0.5)
      {
        tmp1878 += 1.0;
        tmp1877 -= 1.0;
      }
      if(fabs(tmp1878) < 1e-10 && ((unsigned long)tmp1877 & 1))
      {
        tmp1874 = -pow(-tmp1872, tmp1875)*pow(tmp1872, tmp1876);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1872, tmp1873);
      }
    }
  }
  else
  {
    tmp1874 = pow(tmp1872, tmp1873);
  }
  if(isnan(tmp1874) || isinf(tmp1874))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1872, tmp1873);
  }tmp1879 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1874,"(r_init[313] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1879 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[313] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1879);
    }
  }
  (data->simulationInfo->realParameter[817] /* omega_c[313] PARAM */) = sqrt(tmp1879);
  TRACE_POP
}

/*
equation index: 13377
type: SIMPLE_ASSIGN
r_init[312] = r_min + 312.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13377};
  (data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (312.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13378
type: SIMPLE_ASSIGN
omega_c[312] = sqrt(G * Md / (r_init[312] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13378(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13378};
  modelica_real tmp1880;
  modelica_real tmp1881;
  modelica_real tmp1882;
  modelica_real tmp1883;
  modelica_real tmp1884;
  modelica_real tmp1885;
  modelica_real tmp1886;
  modelica_real tmp1887;
  modelica_real tmp1888;
  modelica_real tmp1889;
  tmp1880 = (data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */);
  tmp1881 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1882 = (tmp1880 * tmp1880) + (tmp1881 * tmp1881);
  tmp1883 = 1.5;
  if(tmp1882 < 0.0 && tmp1883 != 0.0)
  {
    tmp1885 = modf(tmp1883, &tmp1886);
    
    if(tmp1885 > 0.5)
    {
      tmp1885 -= 1.0;
      tmp1886 += 1.0;
    }
    else if(tmp1885 < -0.5)
    {
      tmp1885 += 1.0;
      tmp1886 -= 1.0;
    }
    
    if(fabs(tmp1885) < 1e-10)
      tmp1884 = pow(tmp1882, tmp1886);
    else
    {
      tmp1888 = modf(1.0/tmp1883, &tmp1887);
      if(tmp1888 > 0.5)
      {
        tmp1888 -= 1.0;
        tmp1887 += 1.0;
      }
      else if(tmp1888 < -0.5)
      {
        tmp1888 += 1.0;
        tmp1887 -= 1.0;
      }
      if(fabs(tmp1888) < 1e-10 && ((unsigned long)tmp1887 & 1))
      {
        tmp1884 = -pow(-tmp1882, tmp1885)*pow(tmp1882, tmp1886);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1882, tmp1883);
      }
    }
  }
  else
  {
    tmp1884 = pow(tmp1882, tmp1883);
  }
  if(isnan(tmp1884) || isinf(tmp1884))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1882, tmp1883);
  }tmp1889 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1884,"(r_init[312] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1889 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[312] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1889);
    }
  }
  (data->simulationInfo->realParameter[816] /* omega_c[312] PARAM */) = sqrt(tmp1889);
  TRACE_POP
}

/*
equation index: 13379
type: SIMPLE_ASSIGN
r_init[311] = r_min + 311.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13379};
  (data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (311.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13380
type: SIMPLE_ASSIGN
omega_c[311] = sqrt(G * Md / (r_init[311] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13380};
  modelica_real tmp1890;
  modelica_real tmp1891;
  modelica_real tmp1892;
  modelica_real tmp1893;
  modelica_real tmp1894;
  modelica_real tmp1895;
  modelica_real tmp1896;
  modelica_real tmp1897;
  modelica_real tmp1898;
  modelica_real tmp1899;
  tmp1890 = (data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */);
  tmp1891 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1892 = (tmp1890 * tmp1890) + (tmp1891 * tmp1891);
  tmp1893 = 1.5;
  if(tmp1892 < 0.0 && tmp1893 != 0.0)
  {
    tmp1895 = modf(tmp1893, &tmp1896);
    
    if(tmp1895 > 0.5)
    {
      tmp1895 -= 1.0;
      tmp1896 += 1.0;
    }
    else if(tmp1895 < -0.5)
    {
      tmp1895 += 1.0;
      tmp1896 -= 1.0;
    }
    
    if(fabs(tmp1895) < 1e-10)
      tmp1894 = pow(tmp1892, tmp1896);
    else
    {
      tmp1898 = modf(1.0/tmp1893, &tmp1897);
      if(tmp1898 > 0.5)
      {
        tmp1898 -= 1.0;
        tmp1897 += 1.0;
      }
      else if(tmp1898 < -0.5)
      {
        tmp1898 += 1.0;
        tmp1897 -= 1.0;
      }
      if(fabs(tmp1898) < 1e-10 && ((unsigned long)tmp1897 & 1))
      {
        tmp1894 = -pow(-tmp1892, tmp1895)*pow(tmp1892, tmp1896);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1892, tmp1893);
      }
    }
  }
  else
  {
    tmp1894 = pow(tmp1892, tmp1893);
  }
  if(isnan(tmp1894) || isinf(tmp1894))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1892, tmp1893);
  }tmp1899 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1894,"(r_init[311] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1899 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[311] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1899);
    }
  }
  (data->simulationInfo->realParameter[815] /* omega_c[311] PARAM */) = sqrt(tmp1899);
  TRACE_POP
}

/*
equation index: 13381
type: SIMPLE_ASSIGN
r_init[310] = r_min + 310.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13381};
  (data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (310.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13382
type: SIMPLE_ASSIGN
omega_c[310] = sqrt(G * Md / (r_init[310] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13382(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13382};
  modelica_real tmp1900;
  modelica_real tmp1901;
  modelica_real tmp1902;
  modelica_real tmp1903;
  modelica_real tmp1904;
  modelica_real tmp1905;
  modelica_real tmp1906;
  modelica_real tmp1907;
  modelica_real tmp1908;
  modelica_real tmp1909;
  tmp1900 = (data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */);
  tmp1901 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1902 = (tmp1900 * tmp1900) + (tmp1901 * tmp1901);
  tmp1903 = 1.5;
  if(tmp1902 < 0.0 && tmp1903 != 0.0)
  {
    tmp1905 = modf(tmp1903, &tmp1906);
    
    if(tmp1905 > 0.5)
    {
      tmp1905 -= 1.0;
      tmp1906 += 1.0;
    }
    else if(tmp1905 < -0.5)
    {
      tmp1905 += 1.0;
      tmp1906 -= 1.0;
    }
    
    if(fabs(tmp1905) < 1e-10)
      tmp1904 = pow(tmp1902, tmp1906);
    else
    {
      tmp1908 = modf(1.0/tmp1903, &tmp1907);
      if(tmp1908 > 0.5)
      {
        tmp1908 -= 1.0;
        tmp1907 += 1.0;
      }
      else if(tmp1908 < -0.5)
      {
        tmp1908 += 1.0;
        tmp1907 -= 1.0;
      }
      if(fabs(tmp1908) < 1e-10 && ((unsigned long)tmp1907 & 1))
      {
        tmp1904 = -pow(-tmp1902, tmp1905)*pow(tmp1902, tmp1906);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1902, tmp1903);
      }
    }
  }
  else
  {
    tmp1904 = pow(tmp1902, tmp1903);
  }
  if(isnan(tmp1904) || isinf(tmp1904))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1902, tmp1903);
  }tmp1909 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1904,"(r_init[310] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1909 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[310] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1909);
    }
  }
  (data->simulationInfo->realParameter[814] /* omega_c[310] PARAM */) = sqrt(tmp1909);
  TRACE_POP
}

/*
equation index: 13383
type: SIMPLE_ASSIGN
r_init[309] = r_min + 309.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13383};
  (data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (309.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13384
type: SIMPLE_ASSIGN
omega_c[309] = sqrt(G * Md / (r_init[309] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13384(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13384};
  modelica_real tmp1910;
  modelica_real tmp1911;
  modelica_real tmp1912;
  modelica_real tmp1913;
  modelica_real tmp1914;
  modelica_real tmp1915;
  modelica_real tmp1916;
  modelica_real tmp1917;
  modelica_real tmp1918;
  modelica_real tmp1919;
  tmp1910 = (data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */);
  tmp1911 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1912 = (tmp1910 * tmp1910) + (tmp1911 * tmp1911);
  tmp1913 = 1.5;
  if(tmp1912 < 0.0 && tmp1913 != 0.0)
  {
    tmp1915 = modf(tmp1913, &tmp1916);
    
    if(tmp1915 > 0.5)
    {
      tmp1915 -= 1.0;
      tmp1916 += 1.0;
    }
    else if(tmp1915 < -0.5)
    {
      tmp1915 += 1.0;
      tmp1916 -= 1.0;
    }
    
    if(fabs(tmp1915) < 1e-10)
      tmp1914 = pow(tmp1912, tmp1916);
    else
    {
      tmp1918 = modf(1.0/tmp1913, &tmp1917);
      if(tmp1918 > 0.5)
      {
        tmp1918 -= 1.0;
        tmp1917 += 1.0;
      }
      else if(tmp1918 < -0.5)
      {
        tmp1918 += 1.0;
        tmp1917 -= 1.0;
      }
      if(fabs(tmp1918) < 1e-10 && ((unsigned long)tmp1917 & 1))
      {
        tmp1914 = -pow(-tmp1912, tmp1915)*pow(tmp1912, tmp1916);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1912, tmp1913);
      }
    }
  }
  else
  {
    tmp1914 = pow(tmp1912, tmp1913);
  }
  if(isnan(tmp1914) || isinf(tmp1914))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1912, tmp1913);
  }tmp1919 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1914,"(r_init[309] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1919 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[309] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1919);
    }
  }
  (data->simulationInfo->realParameter[813] /* omega_c[309] PARAM */) = sqrt(tmp1919);
  TRACE_POP
}

/*
equation index: 13385
type: SIMPLE_ASSIGN
r_init[308] = r_min + 308.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13385};
  (data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (308.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13386
type: SIMPLE_ASSIGN
omega_c[308] = sqrt(G * Md / (r_init[308] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13386(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13386};
  modelica_real tmp1920;
  modelica_real tmp1921;
  modelica_real tmp1922;
  modelica_real tmp1923;
  modelica_real tmp1924;
  modelica_real tmp1925;
  modelica_real tmp1926;
  modelica_real tmp1927;
  modelica_real tmp1928;
  modelica_real tmp1929;
  tmp1920 = (data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */);
  tmp1921 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1922 = (tmp1920 * tmp1920) + (tmp1921 * tmp1921);
  tmp1923 = 1.5;
  if(tmp1922 < 0.0 && tmp1923 != 0.0)
  {
    tmp1925 = modf(tmp1923, &tmp1926);
    
    if(tmp1925 > 0.5)
    {
      tmp1925 -= 1.0;
      tmp1926 += 1.0;
    }
    else if(tmp1925 < -0.5)
    {
      tmp1925 += 1.0;
      tmp1926 -= 1.0;
    }
    
    if(fabs(tmp1925) < 1e-10)
      tmp1924 = pow(tmp1922, tmp1926);
    else
    {
      tmp1928 = modf(1.0/tmp1923, &tmp1927);
      if(tmp1928 > 0.5)
      {
        tmp1928 -= 1.0;
        tmp1927 += 1.0;
      }
      else if(tmp1928 < -0.5)
      {
        tmp1928 += 1.0;
        tmp1927 -= 1.0;
      }
      if(fabs(tmp1928) < 1e-10 && ((unsigned long)tmp1927 & 1))
      {
        tmp1924 = -pow(-tmp1922, tmp1925)*pow(tmp1922, tmp1926);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1922, tmp1923);
      }
    }
  }
  else
  {
    tmp1924 = pow(tmp1922, tmp1923);
  }
  if(isnan(tmp1924) || isinf(tmp1924))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1922, tmp1923);
  }tmp1929 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1924,"(r_init[308] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1929 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[308] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1929);
    }
  }
  (data->simulationInfo->realParameter[812] /* omega_c[308] PARAM */) = sqrt(tmp1929);
  TRACE_POP
}

/*
equation index: 13387
type: SIMPLE_ASSIGN
r_init[307] = r_min + 307.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13387};
  (data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (307.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13388
type: SIMPLE_ASSIGN
omega_c[307] = sqrt(G * Md / (r_init[307] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13388};
  modelica_real tmp1930;
  modelica_real tmp1931;
  modelica_real tmp1932;
  modelica_real tmp1933;
  modelica_real tmp1934;
  modelica_real tmp1935;
  modelica_real tmp1936;
  modelica_real tmp1937;
  modelica_real tmp1938;
  modelica_real tmp1939;
  tmp1930 = (data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */);
  tmp1931 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1932 = (tmp1930 * tmp1930) + (tmp1931 * tmp1931);
  tmp1933 = 1.5;
  if(tmp1932 < 0.0 && tmp1933 != 0.0)
  {
    tmp1935 = modf(tmp1933, &tmp1936);
    
    if(tmp1935 > 0.5)
    {
      tmp1935 -= 1.0;
      tmp1936 += 1.0;
    }
    else if(tmp1935 < -0.5)
    {
      tmp1935 += 1.0;
      tmp1936 -= 1.0;
    }
    
    if(fabs(tmp1935) < 1e-10)
      tmp1934 = pow(tmp1932, tmp1936);
    else
    {
      tmp1938 = modf(1.0/tmp1933, &tmp1937);
      if(tmp1938 > 0.5)
      {
        tmp1938 -= 1.0;
        tmp1937 += 1.0;
      }
      else if(tmp1938 < -0.5)
      {
        tmp1938 += 1.0;
        tmp1937 -= 1.0;
      }
      if(fabs(tmp1938) < 1e-10 && ((unsigned long)tmp1937 & 1))
      {
        tmp1934 = -pow(-tmp1932, tmp1935)*pow(tmp1932, tmp1936);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1932, tmp1933);
      }
    }
  }
  else
  {
    tmp1934 = pow(tmp1932, tmp1933);
  }
  if(isnan(tmp1934) || isinf(tmp1934))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1932, tmp1933);
  }tmp1939 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1934,"(r_init[307] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1939 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[307] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1939);
    }
  }
  (data->simulationInfo->realParameter[811] /* omega_c[307] PARAM */) = sqrt(tmp1939);
  TRACE_POP
}

/*
equation index: 13389
type: SIMPLE_ASSIGN
r_init[306] = r_min + 306.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13389};
  (data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (306.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13390
type: SIMPLE_ASSIGN
omega_c[306] = sqrt(G * Md / (r_init[306] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13390(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13390};
  modelica_real tmp1940;
  modelica_real tmp1941;
  modelica_real tmp1942;
  modelica_real tmp1943;
  modelica_real tmp1944;
  modelica_real tmp1945;
  modelica_real tmp1946;
  modelica_real tmp1947;
  modelica_real tmp1948;
  modelica_real tmp1949;
  tmp1940 = (data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */);
  tmp1941 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1942 = (tmp1940 * tmp1940) + (tmp1941 * tmp1941);
  tmp1943 = 1.5;
  if(tmp1942 < 0.0 && tmp1943 != 0.0)
  {
    tmp1945 = modf(tmp1943, &tmp1946);
    
    if(tmp1945 > 0.5)
    {
      tmp1945 -= 1.0;
      tmp1946 += 1.0;
    }
    else if(tmp1945 < -0.5)
    {
      tmp1945 += 1.0;
      tmp1946 -= 1.0;
    }
    
    if(fabs(tmp1945) < 1e-10)
      tmp1944 = pow(tmp1942, tmp1946);
    else
    {
      tmp1948 = modf(1.0/tmp1943, &tmp1947);
      if(tmp1948 > 0.5)
      {
        tmp1948 -= 1.0;
        tmp1947 += 1.0;
      }
      else if(tmp1948 < -0.5)
      {
        tmp1948 += 1.0;
        tmp1947 -= 1.0;
      }
      if(fabs(tmp1948) < 1e-10 && ((unsigned long)tmp1947 & 1))
      {
        tmp1944 = -pow(-tmp1942, tmp1945)*pow(tmp1942, tmp1946);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1942, tmp1943);
      }
    }
  }
  else
  {
    tmp1944 = pow(tmp1942, tmp1943);
  }
  if(isnan(tmp1944) || isinf(tmp1944))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1942, tmp1943);
  }tmp1949 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1944,"(r_init[306] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1949 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[306] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1949);
    }
  }
  (data->simulationInfo->realParameter[810] /* omega_c[306] PARAM */) = sqrt(tmp1949);
  TRACE_POP
}

/*
equation index: 13391
type: SIMPLE_ASSIGN
r_init[305] = r_min + 305.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13391};
  (data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (305.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13392
type: SIMPLE_ASSIGN
omega_c[305] = sqrt(G * Md / (r_init[305] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13392(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13392};
  modelica_real tmp1950;
  modelica_real tmp1951;
  modelica_real tmp1952;
  modelica_real tmp1953;
  modelica_real tmp1954;
  modelica_real tmp1955;
  modelica_real tmp1956;
  modelica_real tmp1957;
  modelica_real tmp1958;
  modelica_real tmp1959;
  tmp1950 = (data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */);
  tmp1951 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1952 = (tmp1950 * tmp1950) + (tmp1951 * tmp1951);
  tmp1953 = 1.5;
  if(tmp1952 < 0.0 && tmp1953 != 0.0)
  {
    tmp1955 = modf(tmp1953, &tmp1956);
    
    if(tmp1955 > 0.5)
    {
      tmp1955 -= 1.0;
      tmp1956 += 1.0;
    }
    else if(tmp1955 < -0.5)
    {
      tmp1955 += 1.0;
      tmp1956 -= 1.0;
    }
    
    if(fabs(tmp1955) < 1e-10)
      tmp1954 = pow(tmp1952, tmp1956);
    else
    {
      tmp1958 = modf(1.0/tmp1953, &tmp1957);
      if(tmp1958 > 0.5)
      {
        tmp1958 -= 1.0;
        tmp1957 += 1.0;
      }
      else if(tmp1958 < -0.5)
      {
        tmp1958 += 1.0;
        tmp1957 -= 1.0;
      }
      if(fabs(tmp1958) < 1e-10 && ((unsigned long)tmp1957 & 1))
      {
        tmp1954 = -pow(-tmp1952, tmp1955)*pow(tmp1952, tmp1956);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1952, tmp1953);
      }
    }
  }
  else
  {
    tmp1954 = pow(tmp1952, tmp1953);
  }
  if(isnan(tmp1954) || isinf(tmp1954))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1952, tmp1953);
  }tmp1959 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1954,"(r_init[305] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1959 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[305] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1959);
    }
  }
  (data->simulationInfo->realParameter[809] /* omega_c[305] PARAM */) = sqrt(tmp1959);
  TRACE_POP
}

/*
equation index: 13393
type: SIMPLE_ASSIGN
r_init[304] = r_min + 304.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13393};
  (data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (304.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13394
type: SIMPLE_ASSIGN
omega_c[304] = sqrt(G * Md / (r_init[304] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13394(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13394};
  modelica_real tmp1960;
  modelica_real tmp1961;
  modelica_real tmp1962;
  modelica_real tmp1963;
  modelica_real tmp1964;
  modelica_real tmp1965;
  modelica_real tmp1966;
  modelica_real tmp1967;
  modelica_real tmp1968;
  modelica_real tmp1969;
  tmp1960 = (data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */);
  tmp1961 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1962 = (tmp1960 * tmp1960) + (tmp1961 * tmp1961);
  tmp1963 = 1.5;
  if(tmp1962 < 0.0 && tmp1963 != 0.0)
  {
    tmp1965 = modf(tmp1963, &tmp1966);
    
    if(tmp1965 > 0.5)
    {
      tmp1965 -= 1.0;
      tmp1966 += 1.0;
    }
    else if(tmp1965 < -0.5)
    {
      tmp1965 += 1.0;
      tmp1966 -= 1.0;
    }
    
    if(fabs(tmp1965) < 1e-10)
      tmp1964 = pow(tmp1962, tmp1966);
    else
    {
      tmp1968 = modf(1.0/tmp1963, &tmp1967);
      if(tmp1968 > 0.5)
      {
        tmp1968 -= 1.0;
        tmp1967 += 1.0;
      }
      else if(tmp1968 < -0.5)
      {
        tmp1968 += 1.0;
        tmp1967 -= 1.0;
      }
      if(fabs(tmp1968) < 1e-10 && ((unsigned long)tmp1967 & 1))
      {
        tmp1964 = -pow(-tmp1962, tmp1965)*pow(tmp1962, tmp1966);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1962, tmp1963);
      }
    }
  }
  else
  {
    tmp1964 = pow(tmp1962, tmp1963);
  }
  if(isnan(tmp1964) || isinf(tmp1964))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1962, tmp1963);
  }tmp1969 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1964,"(r_init[304] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1969 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[304] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1969);
    }
  }
  (data->simulationInfo->realParameter[808] /* omega_c[304] PARAM */) = sqrt(tmp1969);
  TRACE_POP
}

/*
equation index: 13395
type: SIMPLE_ASSIGN
r_init[303] = r_min + 303.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13395};
  (data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (303.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13396
type: SIMPLE_ASSIGN
omega_c[303] = sqrt(G * Md / (r_init[303] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13396};
  modelica_real tmp1970;
  modelica_real tmp1971;
  modelica_real tmp1972;
  modelica_real tmp1973;
  modelica_real tmp1974;
  modelica_real tmp1975;
  modelica_real tmp1976;
  modelica_real tmp1977;
  modelica_real tmp1978;
  modelica_real tmp1979;
  tmp1970 = (data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */);
  tmp1971 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1972 = (tmp1970 * tmp1970) + (tmp1971 * tmp1971);
  tmp1973 = 1.5;
  if(tmp1972 < 0.0 && tmp1973 != 0.0)
  {
    tmp1975 = modf(tmp1973, &tmp1976);
    
    if(tmp1975 > 0.5)
    {
      tmp1975 -= 1.0;
      tmp1976 += 1.0;
    }
    else if(tmp1975 < -0.5)
    {
      tmp1975 += 1.0;
      tmp1976 -= 1.0;
    }
    
    if(fabs(tmp1975) < 1e-10)
      tmp1974 = pow(tmp1972, tmp1976);
    else
    {
      tmp1978 = modf(1.0/tmp1973, &tmp1977);
      if(tmp1978 > 0.5)
      {
        tmp1978 -= 1.0;
        tmp1977 += 1.0;
      }
      else if(tmp1978 < -0.5)
      {
        tmp1978 += 1.0;
        tmp1977 -= 1.0;
      }
      if(fabs(tmp1978) < 1e-10 && ((unsigned long)tmp1977 & 1))
      {
        tmp1974 = -pow(-tmp1972, tmp1975)*pow(tmp1972, tmp1976);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1972, tmp1973);
      }
    }
  }
  else
  {
    tmp1974 = pow(tmp1972, tmp1973);
  }
  if(isnan(tmp1974) || isinf(tmp1974))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1972, tmp1973);
  }tmp1979 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1974,"(r_init[303] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1979 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[303] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1979);
    }
  }
  (data->simulationInfo->realParameter[807] /* omega_c[303] PARAM */) = sqrt(tmp1979);
  TRACE_POP
}

/*
equation index: 13397
type: SIMPLE_ASSIGN
r_init[302] = r_min + 302.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13397};
  (data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (302.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13398
type: SIMPLE_ASSIGN
omega_c[302] = sqrt(G * Md / (r_init[302] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13398(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13398};
  modelica_real tmp1980;
  modelica_real tmp1981;
  modelica_real tmp1982;
  modelica_real tmp1983;
  modelica_real tmp1984;
  modelica_real tmp1985;
  modelica_real tmp1986;
  modelica_real tmp1987;
  modelica_real tmp1988;
  modelica_real tmp1989;
  tmp1980 = (data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */);
  tmp1981 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1982 = (tmp1980 * tmp1980) + (tmp1981 * tmp1981);
  tmp1983 = 1.5;
  if(tmp1982 < 0.0 && tmp1983 != 0.0)
  {
    tmp1985 = modf(tmp1983, &tmp1986);
    
    if(tmp1985 > 0.5)
    {
      tmp1985 -= 1.0;
      tmp1986 += 1.0;
    }
    else if(tmp1985 < -0.5)
    {
      tmp1985 += 1.0;
      tmp1986 -= 1.0;
    }
    
    if(fabs(tmp1985) < 1e-10)
      tmp1984 = pow(tmp1982, tmp1986);
    else
    {
      tmp1988 = modf(1.0/tmp1983, &tmp1987);
      if(tmp1988 > 0.5)
      {
        tmp1988 -= 1.0;
        tmp1987 += 1.0;
      }
      else if(tmp1988 < -0.5)
      {
        tmp1988 += 1.0;
        tmp1987 -= 1.0;
      }
      if(fabs(tmp1988) < 1e-10 && ((unsigned long)tmp1987 & 1))
      {
        tmp1984 = -pow(-tmp1982, tmp1985)*pow(tmp1982, tmp1986);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1982, tmp1983);
      }
    }
  }
  else
  {
    tmp1984 = pow(tmp1982, tmp1983);
  }
  if(isnan(tmp1984) || isinf(tmp1984))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1982, tmp1983);
  }tmp1989 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1984,"(r_init[302] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1989 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[302] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1989);
    }
  }
  (data->simulationInfo->realParameter[806] /* omega_c[302] PARAM */) = sqrt(tmp1989);
  TRACE_POP
}

/*
equation index: 13399
type: SIMPLE_ASSIGN
r_init[301] = r_min + 301.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13399};
  (data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (301.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13400
type: SIMPLE_ASSIGN
omega_c[301] = sqrt(G * Md / (r_init[301] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13400(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13400};
  modelica_real tmp1990;
  modelica_real tmp1991;
  modelica_real tmp1992;
  modelica_real tmp1993;
  modelica_real tmp1994;
  modelica_real tmp1995;
  modelica_real tmp1996;
  modelica_real tmp1997;
  modelica_real tmp1998;
  modelica_real tmp1999;
  tmp1990 = (data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */);
  tmp1991 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp1992 = (tmp1990 * tmp1990) + (tmp1991 * tmp1991);
  tmp1993 = 1.5;
  if(tmp1992 < 0.0 && tmp1993 != 0.0)
  {
    tmp1995 = modf(tmp1993, &tmp1996);
    
    if(tmp1995 > 0.5)
    {
      tmp1995 -= 1.0;
      tmp1996 += 1.0;
    }
    else if(tmp1995 < -0.5)
    {
      tmp1995 += 1.0;
      tmp1996 -= 1.0;
    }
    
    if(fabs(tmp1995) < 1e-10)
      tmp1994 = pow(tmp1992, tmp1996);
    else
    {
      tmp1998 = modf(1.0/tmp1993, &tmp1997);
      if(tmp1998 > 0.5)
      {
        tmp1998 -= 1.0;
        tmp1997 += 1.0;
      }
      else if(tmp1998 < -0.5)
      {
        tmp1998 += 1.0;
        tmp1997 -= 1.0;
      }
      if(fabs(tmp1998) < 1e-10 && ((unsigned long)tmp1997 & 1))
      {
        tmp1994 = -pow(-tmp1992, tmp1995)*pow(tmp1992, tmp1996);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1992, tmp1993);
      }
    }
  }
  else
  {
    tmp1994 = pow(tmp1992, tmp1993);
  }
  if(isnan(tmp1994) || isinf(tmp1994))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1992, tmp1993);
  }tmp1999 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp1994,"(r_init[301] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp1999 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[301] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp1999);
    }
  }
  (data->simulationInfo->realParameter[805] /* omega_c[301] PARAM */) = sqrt(tmp1999);
  TRACE_POP
}

/*
equation index: 13401
type: SIMPLE_ASSIGN
r_init[300] = r_min + 300.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13401};
  (data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (300.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13402
type: SIMPLE_ASSIGN
omega_c[300] = sqrt(G * Md / (r_init[300] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13402(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13402};
  modelica_real tmp2000;
  modelica_real tmp2001;
  modelica_real tmp2002;
  modelica_real tmp2003;
  modelica_real tmp2004;
  modelica_real tmp2005;
  modelica_real tmp2006;
  modelica_real tmp2007;
  modelica_real tmp2008;
  modelica_real tmp2009;
  tmp2000 = (data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */);
  tmp2001 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2002 = (tmp2000 * tmp2000) + (tmp2001 * tmp2001);
  tmp2003 = 1.5;
  if(tmp2002 < 0.0 && tmp2003 != 0.0)
  {
    tmp2005 = modf(tmp2003, &tmp2006);
    
    if(tmp2005 > 0.5)
    {
      tmp2005 -= 1.0;
      tmp2006 += 1.0;
    }
    else if(tmp2005 < -0.5)
    {
      tmp2005 += 1.0;
      tmp2006 -= 1.0;
    }
    
    if(fabs(tmp2005) < 1e-10)
      tmp2004 = pow(tmp2002, tmp2006);
    else
    {
      tmp2008 = modf(1.0/tmp2003, &tmp2007);
      if(tmp2008 > 0.5)
      {
        tmp2008 -= 1.0;
        tmp2007 += 1.0;
      }
      else if(tmp2008 < -0.5)
      {
        tmp2008 += 1.0;
        tmp2007 -= 1.0;
      }
      if(fabs(tmp2008) < 1e-10 && ((unsigned long)tmp2007 & 1))
      {
        tmp2004 = -pow(-tmp2002, tmp2005)*pow(tmp2002, tmp2006);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2002, tmp2003);
      }
    }
  }
  else
  {
    tmp2004 = pow(tmp2002, tmp2003);
  }
  if(isnan(tmp2004) || isinf(tmp2004))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2002, tmp2003);
  }tmp2009 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2004,"(r_init[300] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2009 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[300] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2009);
    }
  }
  (data->simulationInfo->realParameter[804] /* omega_c[300] PARAM */) = sqrt(tmp2009);
  TRACE_POP
}

/*
equation index: 13403
type: SIMPLE_ASSIGN
r_init[299] = r_min + 299.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13403};
  (data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (299.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13404
type: SIMPLE_ASSIGN
omega_c[299] = sqrt(G * Md / (r_init[299] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13404};
  modelica_real tmp2010;
  modelica_real tmp2011;
  modelica_real tmp2012;
  modelica_real tmp2013;
  modelica_real tmp2014;
  modelica_real tmp2015;
  modelica_real tmp2016;
  modelica_real tmp2017;
  modelica_real tmp2018;
  modelica_real tmp2019;
  tmp2010 = (data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */);
  tmp2011 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2012 = (tmp2010 * tmp2010) + (tmp2011 * tmp2011);
  tmp2013 = 1.5;
  if(tmp2012 < 0.0 && tmp2013 != 0.0)
  {
    tmp2015 = modf(tmp2013, &tmp2016);
    
    if(tmp2015 > 0.5)
    {
      tmp2015 -= 1.0;
      tmp2016 += 1.0;
    }
    else if(tmp2015 < -0.5)
    {
      tmp2015 += 1.0;
      tmp2016 -= 1.0;
    }
    
    if(fabs(tmp2015) < 1e-10)
      tmp2014 = pow(tmp2012, tmp2016);
    else
    {
      tmp2018 = modf(1.0/tmp2013, &tmp2017);
      if(tmp2018 > 0.5)
      {
        tmp2018 -= 1.0;
        tmp2017 += 1.0;
      }
      else if(tmp2018 < -0.5)
      {
        tmp2018 += 1.0;
        tmp2017 -= 1.0;
      }
      if(fabs(tmp2018) < 1e-10 && ((unsigned long)tmp2017 & 1))
      {
        tmp2014 = -pow(-tmp2012, tmp2015)*pow(tmp2012, tmp2016);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2012, tmp2013);
      }
    }
  }
  else
  {
    tmp2014 = pow(tmp2012, tmp2013);
  }
  if(isnan(tmp2014) || isinf(tmp2014))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2012, tmp2013);
  }tmp2019 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2014,"(r_init[299] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2019 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[299] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2019);
    }
  }
  (data->simulationInfo->realParameter[803] /* omega_c[299] PARAM */) = sqrt(tmp2019);
  TRACE_POP
}

/*
equation index: 13405
type: SIMPLE_ASSIGN
r_init[298] = r_min + 298.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13405};
  (data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (298.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13406
type: SIMPLE_ASSIGN
omega_c[298] = sqrt(G * Md / (r_init[298] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13406(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13406};
  modelica_real tmp2020;
  modelica_real tmp2021;
  modelica_real tmp2022;
  modelica_real tmp2023;
  modelica_real tmp2024;
  modelica_real tmp2025;
  modelica_real tmp2026;
  modelica_real tmp2027;
  modelica_real tmp2028;
  modelica_real tmp2029;
  tmp2020 = (data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */);
  tmp2021 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2022 = (tmp2020 * tmp2020) + (tmp2021 * tmp2021);
  tmp2023 = 1.5;
  if(tmp2022 < 0.0 && tmp2023 != 0.0)
  {
    tmp2025 = modf(tmp2023, &tmp2026);
    
    if(tmp2025 > 0.5)
    {
      tmp2025 -= 1.0;
      tmp2026 += 1.0;
    }
    else if(tmp2025 < -0.5)
    {
      tmp2025 += 1.0;
      tmp2026 -= 1.0;
    }
    
    if(fabs(tmp2025) < 1e-10)
      tmp2024 = pow(tmp2022, tmp2026);
    else
    {
      tmp2028 = modf(1.0/tmp2023, &tmp2027);
      if(tmp2028 > 0.5)
      {
        tmp2028 -= 1.0;
        tmp2027 += 1.0;
      }
      else if(tmp2028 < -0.5)
      {
        tmp2028 += 1.0;
        tmp2027 -= 1.0;
      }
      if(fabs(tmp2028) < 1e-10 && ((unsigned long)tmp2027 & 1))
      {
        tmp2024 = -pow(-tmp2022, tmp2025)*pow(tmp2022, tmp2026);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2022, tmp2023);
      }
    }
  }
  else
  {
    tmp2024 = pow(tmp2022, tmp2023);
  }
  if(isnan(tmp2024) || isinf(tmp2024))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2022, tmp2023);
  }tmp2029 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2024,"(r_init[298] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2029 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[298] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2029);
    }
  }
  (data->simulationInfo->realParameter[802] /* omega_c[298] PARAM */) = sqrt(tmp2029);
  TRACE_POP
}

/*
equation index: 13407
type: SIMPLE_ASSIGN
r_init[297] = r_min + 297.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13407};
  (data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (297.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13408
type: SIMPLE_ASSIGN
omega_c[297] = sqrt(G * Md / (r_init[297] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13408(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13408};
  modelica_real tmp2030;
  modelica_real tmp2031;
  modelica_real tmp2032;
  modelica_real tmp2033;
  modelica_real tmp2034;
  modelica_real tmp2035;
  modelica_real tmp2036;
  modelica_real tmp2037;
  modelica_real tmp2038;
  modelica_real tmp2039;
  tmp2030 = (data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */);
  tmp2031 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2032 = (tmp2030 * tmp2030) + (tmp2031 * tmp2031);
  tmp2033 = 1.5;
  if(tmp2032 < 0.0 && tmp2033 != 0.0)
  {
    tmp2035 = modf(tmp2033, &tmp2036);
    
    if(tmp2035 > 0.5)
    {
      tmp2035 -= 1.0;
      tmp2036 += 1.0;
    }
    else if(tmp2035 < -0.5)
    {
      tmp2035 += 1.0;
      tmp2036 -= 1.0;
    }
    
    if(fabs(tmp2035) < 1e-10)
      tmp2034 = pow(tmp2032, tmp2036);
    else
    {
      tmp2038 = modf(1.0/tmp2033, &tmp2037);
      if(tmp2038 > 0.5)
      {
        tmp2038 -= 1.0;
        tmp2037 += 1.0;
      }
      else if(tmp2038 < -0.5)
      {
        tmp2038 += 1.0;
        tmp2037 -= 1.0;
      }
      if(fabs(tmp2038) < 1e-10 && ((unsigned long)tmp2037 & 1))
      {
        tmp2034 = -pow(-tmp2032, tmp2035)*pow(tmp2032, tmp2036);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2032, tmp2033);
      }
    }
  }
  else
  {
    tmp2034 = pow(tmp2032, tmp2033);
  }
  if(isnan(tmp2034) || isinf(tmp2034))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2032, tmp2033);
  }tmp2039 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2034,"(r_init[297] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2039 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[297] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2039);
    }
  }
  (data->simulationInfo->realParameter[801] /* omega_c[297] PARAM */) = sqrt(tmp2039);
  TRACE_POP
}

/*
equation index: 13409
type: SIMPLE_ASSIGN
r_init[296] = r_min + 296.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13409};
  (data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (296.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13410
type: SIMPLE_ASSIGN
omega_c[296] = sqrt(G * Md / (r_init[296] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13410(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13410};
  modelica_real tmp2040;
  modelica_real tmp2041;
  modelica_real tmp2042;
  modelica_real tmp2043;
  modelica_real tmp2044;
  modelica_real tmp2045;
  modelica_real tmp2046;
  modelica_real tmp2047;
  modelica_real tmp2048;
  modelica_real tmp2049;
  tmp2040 = (data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */);
  tmp2041 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2042 = (tmp2040 * tmp2040) + (tmp2041 * tmp2041);
  tmp2043 = 1.5;
  if(tmp2042 < 0.0 && tmp2043 != 0.0)
  {
    tmp2045 = modf(tmp2043, &tmp2046);
    
    if(tmp2045 > 0.5)
    {
      tmp2045 -= 1.0;
      tmp2046 += 1.0;
    }
    else if(tmp2045 < -0.5)
    {
      tmp2045 += 1.0;
      tmp2046 -= 1.0;
    }
    
    if(fabs(tmp2045) < 1e-10)
      tmp2044 = pow(tmp2042, tmp2046);
    else
    {
      tmp2048 = modf(1.0/tmp2043, &tmp2047);
      if(tmp2048 > 0.5)
      {
        tmp2048 -= 1.0;
        tmp2047 += 1.0;
      }
      else if(tmp2048 < -0.5)
      {
        tmp2048 += 1.0;
        tmp2047 -= 1.0;
      }
      if(fabs(tmp2048) < 1e-10 && ((unsigned long)tmp2047 & 1))
      {
        tmp2044 = -pow(-tmp2042, tmp2045)*pow(tmp2042, tmp2046);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2042, tmp2043);
      }
    }
  }
  else
  {
    tmp2044 = pow(tmp2042, tmp2043);
  }
  if(isnan(tmp2044) || isinf(tmp2044))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2042, tmp2043);
  }tmp2049 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2044,"(r_init[296] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2049 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[296] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2049);
    }
  }
  (data->simulationInfo->realParameter[800] /* omega_c[296] PARAM */) = sqrt(tmp2049);
  TRACE_POP
}

/*
equation index: 13411
type: SIMPLE_ASSIGN
r_init[295] = r_min + 295.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13411};
  (data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (295.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13412
type: SIMPLE_ASSIGN
omega_c[295] = sqrt(G * Md / (r_init[295] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13412};
  modelica_real tmp2050;
  modelica_real tmp2051;
  modelica_real tmp2052;
  modelica_real tmp2053;
  modelica_real tmp2054;
  modelica_real tmp2055;
  modelica_real tmp2056;
  modelica_real tmp2057;
  modelica_real tmp2058;
  modelica_real tmp2059;
  tmp2050 = (data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */);
  tmp2051 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2052 = (tmp2050 * tmp2050) + (tmp2051 * tmp2051);
  tmp2053 = 1.5;
  if(tmp2052 < 0.0 && tmp2053 != 0.0)
  {
    tmp2055 = modf(tmp2053, &tmp2056);
    
    if(tmp2055 > 0.5)
    {
      tmp2055 -= 1.0;
      tmp2056 += 1.0;
    }
    else if(tmp2055 < -0.5)
    {
      tmp2055 += 1.0;
      tmp2056 -= 1.0;
    }
    
    if(fabs(tmp2055) < 1e-10)
      tmp2054 = pow(tmp2052, tmp2056);
    else
    {
      tmp2058 = modf(1.0/tmp2053, &tmp2057);
      if(tmp2058 > 0.5)
      {
        tmp2058 -= 1.0;
        tmp2057 += 1.0;
      }
      else if(tmp2058 < -0.5)
      {
        tmp2058 += 1.0;
        tmp2057 -= 1.0;
      }
      if(fabs(tmp2058) < 1e-10 && ((unsigned long)tmp2057 & 1))
      {
        tmp2054 = -pow(-tmp2052, tmp2055)*pow(tmp2052, tmp2056);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2052, tmp2053);
      }
    }
  }
  else
  {
    tmp2054 = pow(tmp2052, tmp2053);
  }
  if(isnan(tmp2054) || isinf(tmp2054))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2052, tmp2053);
  }tmp2059 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2054,"(r_init[295] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2059 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[295] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2059);
    }
  }
  (data->simulationInfo->realParameter[799] /* omega_c[295] PARAM */) = sqrt(tmp2059);
  TRACE_POP
}

/*
equation index: 13413
type: SIMPLE_ASSIGN
r_init[294] = r_min + 294.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13413};
  (data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (294.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13414
type: SIMPLE_ASSIGN
omega_c[294] = sqrt(G * Md / (r_init[294] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13414(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13414};
  modelica_real tmp2060;
  modelica_real tmp2061;
  modelica_real tmp2062;
  modelica_real tmp2063;
  modelica_real tmp2064;
  modelica_real tmp2065;
  modelica_real tmp2066;
  modelica_real tmp2067;
  modelica_real tmp2068;
  modelica_real tmp2069;
  tmp2060 = (data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */);
  tmp2061 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2062 = (tmp2060 * tmp2060) + (tmp2061 * tmp2061);
  tmp2063 = 1.5;
  if(tmp2062 < 0.0 && tmp2063 != 0.0)
  {
    tmp2065 = modf(tmp2063, &tmp2066);
    
    if(tmp2065 > 0.5)
    {
      tmp2065 -= 1.0;
      tmp2066 += 1.0;
    }
    else if(tmp2065 < -0.5)
    {
      tmp2065 += 1.0;
      tmp2066 -= 1.0;
    }
    
    if(fabs(tmp2065) < 1e-10)
      tmp2064 = pow(tmp2062, tmp2066);
    else
    {
      tmp2068 = modf(1.0/tmp2063, &tmp2067);
      if(tmp2068 > 0.5)
      {
        tmp2068 -= 1.0;
        tmp2067 += 1.0;
      }
      else if(tmp2068 < -0.5)
      {
        tmp2068 += 1.0;
        tmp2067 -= 1.0;
      }
      if(fabs(tmp2068) < 1e-10 && ((unsigned long)tmp2067 & 1))
      {
        tmp2064 = -pow(-tmp2062, tmp2065)*pow(tmp2062, tmp2066);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2062, tmp2063);
      }
    }
  }
  else
  {
    tmp2064 = pow(tmp2062, tmp2063);
  }
  if(isnan(tmp2064) || isinf(tmp2064))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2062, tmp2063);
  }tmp2069 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2064,"(r_init[294] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2069 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[294] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2069);
    }
  }
  (data->simulationInfo->realParameter[798] /* omega_c[294] PARAM */) = sqrt(tmp2069);
  TRACE_POP
}

/*
equation index: 13415
type: SIMPLE_ASSIGN
r_init[293] = r_min + 293.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13415};
  (data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (293.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13416
type: SIMPLE_ASSIGN
omega_c[293] = sqrt(G * Md / (r_init[293] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13416(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13416};
  modelica_real tmp2070;
  modelica_real tmp2071;
  modelica_real tmp2072;
  modelica_real tmp2073;
  modelica_real tmp2074;
  modelica_real tmp2075;
  modelica_real tmp2076;
  modelica_real tmp2077;
  modelica_real tmp2078;
  modelica_real tmp2079;
  tmp2070 = (data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */);
  tmp2071 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2072 = (tmp2070 * tmp2070) + (tmp2071 * tmp2071);
  tmp2073 = 1.5;
  if(tmp2072 < 0.0 && tmp2073 != 0.0)
  {
    tmp2075 = modf(tmp2073, &tmp2076);
    
    if(tmp2075 > 0.5)
    {
      tmp2075 -= 1.0;
      tmp2076 += 1.0;
    }
    else if(tmp2075 < -0.5)
    {
      tmp2075 += 1.0;
      tmp2076 -= 1.0;
    }
    
    if(fabs(tmp2075) < 1e-10)
      tmp2074 = pow(tmp2072, tmp2076);
    else
    {
      tmp2078 = modf(1.0/tmp2073, &tmp2077);
      if(tmp2078 > 0.5)
      {
        tmp2078 -= 1.0;
        tmp2077 += 1.0;
      }
      else if(tmp2078 < -0.5)
      {
        tmp2078 += 1.0;
        tmp2077 -= 1.0;
      }
      if(fabs(tmp2078) < 1e-10 && ((unsigned long)tmp2077 & 1))
      {
        tmp2074 = -pow(-tmp2072, tmp2075)*pow(tmp2072, tmp2076);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2072, tmp2073);
      }
    }
  }
  else
  {
    tmp2074 = pow(tmp2072, tmp2073);
  }
  if(isnan(tmp2074) || isinf(tmp2074))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2072, tmp2073);
  }tmp2079 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2074,"(r_init[293] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2079 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[293] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2079);
    }
  }
  (data->simulationInfo->realParameter[797] /* omega_c[293] PARAM */) = sqrt(tmp2079);
  TRACE_POP
}

/*
equation index: 13417
type: SIMPLE_ASSIGN
r_init[292] = r_min + 292.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13417};
  (data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (292.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13418
type: SIMPLE_ASSIGN
omega_c[292] = sqrt(G * Md / (r_init[292] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13418(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13418};
  modelica_real tmp2080;
  modelica_real tmp2081;
  modelica_real tmp2082;
  modelica_real tmp2083;
  modelica_real tmp2084;
  modelica_real tmp2085;
  modelica_real tmp2086;
  modelica_real tmp2087;
  modelica_real tmp2088;
  modelica_real tmp2089;
  tmp2080 = (data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */);
  tmp2081 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2082 = (tmp2080 * tmp2080) + (tmp2081 * tmp2081);
  tmp2083 = 1.5;
  if(tmp2082 < 0.0 && tmp2083 != 0.0)
  {
    tmp2085 = modf(tmp2083, &tmp2086);
    
    if(tmp2085 > 0.5)
    {
      tmp2085 -= 1.0;
      tmp2086 += 1.0;
    }
    else if(tmp2085 < -0.5)
    {
      tmp2085 += 1.0;
      tmp2086 -= 1.0;
    }
    
    if(fabs(tmp2085) < 1e-10)
      tmp2084 = pow(tmp2082, tmp2086);
    else
    {
      tmp2088 = modf(1.0/tmp2083, &tmp2087);
      if(tmp2088 > 0.5)
      {
        tmp2088 -= 1.0;
        tmp2087 += 1.0;
      }
      else if(tmp2088 < -0.5)
      {
        tmp2088 += 1.0;
        tmp2087 -= 1.0;
      }
      if(fabs(tmp2088) < 1e-10 && ((unsigned long)tmp2087 & 1))
      {
        tmp2084 = -pow(-tmp2082, tmp2085)*pow(tmp2082, tmp2086);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2082, tmp2083);
      }
    }
  }
  else
  {
    tmp2084 = pow(tmp2082, tmp2083);
  }
  if(isnan(tmp2084) || isinf(tmp2084))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2082, tmp2083);
  }tmp2089 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2084,"(r_init[292] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2089 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[292] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2089);
    }
  }
  (data->simulationInfo->realParameter[796] /* omega_c[292] PARAM */) = sqrt(tmp2089);
  TRACE_POP
}

/*
equation index: 13419
type: SIMPLE_ASSIGN
r_init[291] = r_min + 291.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13419};
  (data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (291.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13420
type: SIMPLE_ASSIGN
omega_c[291] = sqrt(G * Md / (r_init[291] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13420};
  modelica_real tmp2090;
  modelica_real tmp2091;
  modelica_real tmp2092;
  modelica_real tmp2093;
  modelica_real tmp2094;
  modelica_real tmp2095;
  modelica_real tmp2096;
  modelica_real tmp2097;
  modelica_real tmp2098;
  modelica_real tmp2099;
  tmp2090 = (data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */);
  tmp2091 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2092 = (tmp2090 * tmp2090) + (tmp2091 * tmp2091);
  tmp2093 = 1.5;
  if(tmp2092 < 0.0 && tmp2093 != 0.0)
  {
    tmp2095 = modf(tmp2093, &tmp2096);
    
    if(tmp2095 > 0.5)
    {
      tmp2095 -= 1.0;
      tmp2096 += 1.0;
    }
    else if(tmp2095 < -0.5)
    {
      tmp2095 += 1.0;
      tmp2096 -= 1.0;
    }
    
    if(fabs(tmp2095) < 1e-10)
      tmp2094 = pow(tmp2092, tmp2096);
    else
    {
      tmp2098 = modf(1.0/tmp2093, &tmp2097);
      if(tmp2098 > 0.5)
      {
        tmp2098 -= 1.0;
        tmp2097 += 1.0;
      }
      else if(tmp2098 < -0.5)
      {
        tmp2098 += 1.0;
        tmp2097 -= 1.0;
      }
      if(fabs(tmp2098) < 1e-10 && ((unsigned long)tmp2097 & 1))
      {
        tmp2094 = -pow(-tmp2092, tmp2095)*pow(tmp2092, tmp2096);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2092, tmp2093);
      }
    }
  }
  else
  {
    tmp2094 = pow(tmp2092, tmp2093);
  }
  if(isnan(tmp2094) || isinf(tmp2094))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2092, tmp2093);
  }tmp2099 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2094,"(r_init[291] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2099 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[291] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2099);
    }
  }
  (data->simulationInfo->realParameter[795] /* omega_c[291] PARAM */) = sqrt(tmp2099);
  TRACE_POP
}

/*
equation index: 13421
type: SIMPLE_ASSIGN
r_init[290] = r_min + 290.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13421};
  (data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (290.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13422
type: SIMPLE_ASSIGN
omega_c[290] = sqrt(G * Md / (r_init[290] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13422(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13422};
  modelica_real tmp2100;
  modelica_real tmp2101;
  modelica_real tmp2102;
  modelica_real tmp2103;
  modelica_real tmp2104;
  modelica_real tmp2105;
  modelica_real tmp2106;
  modelica_real tmp2107;
  modelica_real tmp2108;
  modelica_real tmp2109;
  tmp2100 = (data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */);
  tmp2101 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2102 = (tmp2100 * tmp2100) + (tmp2101 * tmp2101);
  tmp2103 = 1.5;
  if(tmp2102 < 0.0 && tmp2103 != 0.0)
  {
    tmp2105 = modf(tmp2103, &tmp2106);
    
    if(tmp2105 > 0.5)
    {
      tmp2105 -= 1.0;
      tmp2106 += 1.0;
    }
    else if(tmp2105 < -0.5)
    {
      tmp2105 += 1.0;
      tmp2106 -= 1.0;
    }
    
    if(fabs(tmp2105) < 1e-10)
      tmp2104 = pow(tmp2102, tmp2106);
    else
    {
      tmp2108 = modf(1.0/tmp2103, &tmp2107);
      if(tmp2108 > 0.5)
      {
        tmp2108 -= 1.0;
        tmp2107 += 1.0;
      }
      else if(tmp2108 < -0.5)
      {
        tmp2108 += 1.0;
        tmp2107 -= 1.0;
      }
      if(fabs(tmp2108) < 1e-10 && ((unsigned long)tmp2107 & 1))
      {
        tmp2104 = -pow(-tmp2102, tmp2105)*pow(tmp2102, tmp2106);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2102, tmp2103);
      }
    }
  }
  else
  {
    tmp2104 = pow(tmp2102, tmp2103);
  }
  if(isnan(tmp2104) || isinf(tmp2104))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2102, tmp2103);
  }tmp2109 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2104,"(r_init[290] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2109 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[290] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2109);
    }
  }
  (data->simulationInfo->realParameter[794] /* omega_c[290] PARAM */) = sqrt(tmp2109);
  TRACE_POP
}

/*
equation index: 13423
type: SIMPLE_ASSIGN
r_init[289] = r_min + 289.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13423};
  (data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (289.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13424
type: SIMPLE_ASSIGN
omega_c[289] = sqrt(G * Md / (r_init[289] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13424(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13424};
  modelica_real tmp2110;
  modelica_real tmp2111;
  modelica_real tmp2112;
  modelica_real tmp2113;
  modelica_real tmp2114;
  modelica_real tmp2115;
  modelica_real tmp2116;
  modelica_real tmp2117;
  modelica_real tmp2118;
  modelica_real tmp2119;
  tmp2110 = (data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */);
  tmp2111 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2112 = (tmp2110 * tmp2110) + (tmp2111 * tmp2111);
  tmp2113 = 1.5;
  if(tmp2112 < 0.0 && tmp2113 != 0.0)
  {
    tmp2115 = modf(tmp2113, &tmp2116);
    
    if(tmp2115 > 0.5)
    {
      tmp2115 -= 1.0;
      tmp2116 += 1.0;
    }
    else if(tmp2115 < -0.5)
    {
      tmp2115 += 1.0;
      tmp2116 -= 1.0;
    }
    
    if(fabs(tmp2115) < 1e-10)
      tmp2114 = pow(tmp2112, tmp2116);
    else
    {
      tmp2118 = modf(1.0/tmp2113, &tmp2117);
      if(tmp2118 > 0.5)
      {
        tmp2118 -= 1.0;
        tmp2117 += 1.0;
      }
      else if(tmp2118 < -0.5)
      {
        tmp2118 += 1.0;
        tmp2117 -= 1.0;
      }
      if(fabs(tmp2118) < 1e-10 && ((unsigned long)tmp2117 & 1))
      {
        tmp2114 = -pow(-tmp2112, tmp2115)*pow(tmp2112, tmp2116);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2112, tmp2113);
      }
    }
  }
  else
  {
    tmp2114 = pow(tmp2112, tmp2113);
  }
  if(isnan(tmp2114) || isinf(tmp2114))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2112, tmp2113);
  }tmp2119 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2114,"(r_init[289] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2119 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[289] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2119);
    }
  }
  (data->simulationInfo->realParameter[793] /* omega_c[289] PARAM */) = sqrt(tmp2119);
  TRACE_POP
}

/*
equation index: 13425
type: SIMPLE_ASSIGN
r_init[288] = r_min + 288.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13425};
  (data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (288.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13426
type: SIMPLE_ASSIGN
omega_c[288] = sqrt(G * Md / (r_init[288] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13426(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13426};
  modelica_real tmp2120;
  modelica_real tmp2121;
  modelica_real tmp2122;
  modelica_real tmp2123;
  modelica_real tmp2124;
  modelica_real tmp2125;
  modelica_real tmp2126;
  modelica_real tmp2127;
  modelica_real tmp2128;
  modelica_real tmp2129;
  tmp2120 = (data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */);
  tmp2121 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2122 = (tmp2120 * tmp2120) + (tmp2121 * tmp2121);
  tmp2123 = 1.5;
  if(tmp2122 < 0.0 && tmp2123 != 0.0)
  {
    tmp2125 = modf(tmp2123, &tmp2126);
    
    if(tmp2125 > 0.5)
    {
      tmp2125 -= 1.0;
      tmp2126 += 1.0;
    }
    else if(tmp2125 < -0.5)
    {
      tmp2125 += 1.0;
      tmp2126 -= 1.0;
    }
    
    if(fabs(tmp2125) < 1e-10)
      tmp2124 = pow(tmp2122, tmp2126);
    else
    {
      tmp2128 = modf(1.0/tmp2123, &tmp2127);
      if(tmp2128 > 0.5)
      {
        tmp2128 -= 1.0;
        tmp2127 += 1.0;
      }
      else if(tmp2128 < -0.5)
      {
        tmp2128 += 1.0;
        tmp2127 -= 1.0;
      }
      if(fabs(tmp2128) < 1e-10 && ((unsigned long)tmp2127 & 1))
      {
        tmp2124 = -pow(-tmp2122, tmp2125)*pow(tmp2122, tmp2126);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2122, tmp2123);
      }
    }
  }
  else
  {
    tmp2124 = pow(tmp2122, tmp2123);
  }
  if(isnan(tmp2124) || isinf(tmp2124))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2122, tmp2123);
  }tmp2129 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2124,"(r_init[288] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2129 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[288] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2129);
    }
  }
  (data->simulationInfo->realParameter[792] /* omega_c[288] PARAM */) = sqrt(tmp2129);
  TRACE_POP
}

/*
equation index: 13427
type: SIMPLE_ASSIGN
r_init[287] = r_min + 287.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13427};
  (data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (287.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13428
type: SIMPLE_ASSIGN
omega_c[287] = sqrt(G * Md / (r_init[287] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13428};
  modelica_real tmp2130;
  modelica_real tmp2131;
  modelica_real tmp2132;
  modelica_real tmp2133;
  modelica_real tmp2134;
  modelica_real tmp2135;
  modelica_real tmp2136;
  modelica_real tmp2137;
  modelica_real tmp2138;
  modelica_real tmp2139;
  tmp2130 = (data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */);
  tmp2131 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2132 = (tmp2130 * tmp2130) + (tmp2131 * tmp2131);
  tmp2133 = 1.5;
  if(tmp2132 < 0.0 && tmp2133 != 0.0)
  {
    tmp2135 = modf(tmp2133, &tmp2136);
    
    if(tmp2135 > 0.5)
    {
      tmp2135 -= 1.0;
      tmp2136 += 1.0;
    }
    else if(tmp2135 < -0.5)
    {
      tmp2135 += 1.0;
      tmp2136 -= 1.0;
    }
    
    if(fabs(tmp2135) < 1e-10)
      tmp2134 = pow(tmp2132, tmp2136);
    else
    {
      tmp2138 = modf(1.0/tmp2133, &tmp2137);
      if(tmp2138 > 0.5)
      {
        tmp2138 -= 1.0;
        tmp2137 += 1.0;
      }
      else if(tmp2138 < -0.5)
      {
        tmp2138 += 1.0;
        tmp2137 -= 1.0;
      }
      if(fabs(tmp2138) < 1e-10 && ((unsigned long)tmp2137 & 1))
      {
        tmp2134 = -pow(-tmp2132, tmp2135)*pow(tmp2132, tmp2136);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2132, tmp2133);
      }
    }
  }
  else
  {
    tmp2134 = pow(tmp2132, tmp2133);
  }
  if(isnan(tmp2134) || isinf(tmp2134))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2132, tmp2133);
  }tmp2139 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2134,"(r_init[287] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2139 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[287] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2139);
    }
  }
  (data->simulationInfo->realParameter[791] /* omega_c[287] PARAM */) = sqrt(tmp2139);
  TRACE_POP
}

/*
equation index: 13429
type: SIMPLE_ASSIGN
r_init[286] = r_min + 286.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13429};
  (data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (286.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13430
type: SIMPLE_ASSIGN
omega_c[286] = sqrt(G * Md / (r_init[286] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13430(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13430};
  modelica_real tmp2140;
  modelica_real tmp2141;
  modelica_real tmp2142;
  modelica_real tmp2143;
  modelica_real tmp2144;
  modelica_real tmp2145;
  modelica_real tmp2146;
  modelica_real tmp2147;
  modelica_real tmp2148;
  modelica_real tmp2149;
  tmp2140 = (data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */);
  tmp2141 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2142 = (tmp2140 * tmp2140) + (tmp2141 * tmp2141);
  tmp2143 = 1.5;
  if(tmp2142 < 0.0 && tmp2143 != 0.0)
  {
    tmp2145 = modf(tmp2143, &tmp2146);
    
    if(tmp2145 > 0.5)
    {
      tmp2145 -= 1.0;
      tmp2146 += 1.0;
    }
    else if(tmp2145 < -0.5)
    {
      tmp2145 += 1.0;
      tmp2146 -= 1.0;
    }
    
    if(fabs(tmp2145) < 1e-10)
      tmp2144 = pow(tmp2142, tmp2146);
    else
    {
      tmp2148 = modf(1.0/tmp2143, &tmp2147);
      if(tmp2148 > 0.5)
      {
        tmp2148 -= 1.0;
        tmp2147 += 1.0;
      }
      else if(tmp2148 < -0.5)
      {
        tmp2148 += 1.0;
        tmp2147 -= 1.0;
      }
      if(fabs(tmp2148) < 1e-10 && ((unsigned long)tmp2147 & 1))
      {
        tmp2144 = -pow(-tmp2142, tmp2145)*pow(tmp2142, tmp2146);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2142, tmp2143);
      }
    }
  }
  else
  {
    tmp2144 = pow(tmp2142, tmp2143);
  }
  if(isnan(tmp2144) || isinf(tmp2144))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2142, tmp2143);
  }tmp2149 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2144,"(r_init[286] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2149 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[286] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2149);
    }
  }
  (data->simulationInfo->realParameter[790] /* omega_c[286] PARAM */) = sqrt(tmp2149);
  TRACE_POP
}

/*
equation index: 13431
type: SIMPLE_ASSIGN
r_init[285] = r_min + 285.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13431};
  (data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (285.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13432
type: SIMPLE_ASSIGN
omega_c[285] = sqrt(G * Md / (r_init[285] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13432(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13432};
  modelica_real tmp2150;
  modelica_real tmp2151;
  modelica_real tmp2152;
  modelica_real tmp2153;
  modelica_real tmp2154;
  modelica_real tmp2155;
  modelica_real tmp2156;
  modelica_real tmp2157;
  modelica_real tmp2158;
  modelica_real tmp2159;
  tmp2150 = (data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */);
  tmp2151 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2152 = (tmp2150 * tmp2150) + (tmp2151 * tmp2151);
  tmp2153 = 1.5;
  if(tmp2152 < 0.0 && tmp2153 != 0.0)
  {
    tmp2155 = modf(tmp2153, &tmp2156);
    
    if(tmp2155 > 0.5)
    {
      tmp2155 -= 1.0;
      tmp2156 += 1.0;
    }
    else if(tmp2155 < -0.5)
    {
      tmp2155 += 1.0;
      tmp2156 -= 1.0;
    }
    
    if(fabs(tmp2155) < 1e-10)
      tmp2154 = pow(tmp2152, tmp2156);
    else
    {
      tmp2158 = modf(1.0/tmp2153, &tmp2157);
      if(tmp2158 > 0.5)
      {
        tmp2158 -= 1.0;
        tmp2157 += 1.0;
      }
      else if(tmp2158 < -0.5)
      {
        tmp2158 += 1.0;
        tmp2157 -= 1.0;
      }
      if(fabs(tmp2158) < 1e-10 && ((unsigned long)tmp2157 & 1))
      {
        tmp2154 = -pow(-tmp2152, tmp2155)*pow(tmp2152, tmp2156);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2152, tmp2153);
      }
    }
  }
  else
  {
    tmp2154 = pow(tmp2152, tmp2153);
  }
  if(isnan(tmp2154) || isinf(tmp2154))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2152, tmp2153);
  }tmp2159 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2154,"(r_init[285] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2159 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[285] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2159);
    }
  }
  (data->simulationInfo->realParameter[789] /* omega_c[285] PARAM */) = sqrt(tmp2159);
  TRACE_POP
}

/*
equation index: 13433
type: SIMPLE_ASSIGN
r_init[284] = r_min + 284.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13433};
  (data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (284.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13434
type: SIMPLE_ASSIGN
omega_c[284] = sqrt(G * Md / (r_init[284] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13434(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13434};
  modelica_real tmp2160;
  modelica_real tmp2161;
  modelica_real tmp2162;
  modelica_real tmp2163;
  modelica_real tmp2164;
  modelica_real tmp2165;
  modelica_real tmp2166;
  modelica_real tmp2167;
  modelica_real tmp2168;
  modelica_real tmp2169;
  tmp2160 = (data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */);
  tmp2161 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2162 = (tmp2160 * tmp2160) + (tmp2161 * tmp2161);
  tmp2163 = 1.5;
  if(tmp2162 < 0.0 && tmp2163 != 0.0)
  {
    tmp2165 = modf(tmp2163, &tmp2166);
    
    if(tmp2165 > 0.5)
    {
      tmp2165 -= 1.0;
      tmp2166 += 1.0;
    }
    else if(tmp2165 < -0.5)
    {
      tmp2165 += 1.0;
      tmp2166 -= 1.0;
    }
    
    if(fabs(tmp2165) < 1e-10)
      tmp2164 = pow(tmp2162, tmp2166);
    else
    {
      tmp2168 = modf(1.0/tmp2163, &tmp2167);
      if(tmp2168 > 0.5)
      {
        tmp2168 -= 1.0;
        tmp2167 += 1.0;
      }
      else if(tmp2168 < -0.5)
      {
        tmp2168 += 1.0;
        tmp2167 -= 1.0;
      }
      if(fabs(tmp2168) < 1e-10 && ((unsigned long)tmp2167 & 1))
      {
        tmp2164 = -pow(-tmp2162, tmp2165)*pow(tmp2162, tmp2166);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2162, tmp2163);
      }
    }
  }
  else
  {
    tmp2164 = pow(tmp2162, tmp2163);
  }
  if(isnan(tmp2164) || isinf(tmp2164))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2162, tmp2163);
  }tmp2169 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2164,"(r_init[284] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2169 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[284] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2169);
    }
  }
  (data->simulationInfo->realParameter[788] /* omega_c[284] PARAM */) = sqrt(tmp2169);
  TRACE_POP
}

/*
equation index: 13435
type: SIMPLE_ASSIGN
r_init[283] = r_min + 283.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13435};
  (data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (283.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13436
type: SIMPLE_ASSIGN
omega_c[283] = sqrt(G * Md / (r_init[283] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13436};
  modelica_real tmp2170;
  modelica_real tmp2171;
  modelica_real tmp2172;
  modelica_real tmp2173;
  modelica_real tmp2174;
  modelica_real tmp2175;
  modelica_real tmp2176;
  modelica_real tmp2177;
  modelica_real tmp2178;
  modelica_real tmp2179;
  tmp2170 = (data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */);
  tmp2171 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2172 = (tmp2170 * tmp2170) + (tmp2171 * tmp2171);
  tmp2173 = 1.5;
  if(tmp2172 < 0.0 && tmp2173 != 0.0)
  {
    tmp2175 = modf(tmp2173, &tmp2176);
    
    if(tmp2175 > 0.5)
    {
      tmp2175 -= 1.0;
      tmp2176 += 1.0;
    }
    else if(tmp2175 < -0.5)
    {
      tmp2175 += 1.0;
      tmp2176 -= 1.0;
    }
    
    if(fabs(tmp2175) < 1e-10)
      tmp2174 = pow(tmp2172, tmp2176);
    else
    {
      tmp2178 = modf(1.0/tmp2173, &tmp2177);
      if(tmp2178 > 0.5)
      {
        tmp2178 -= 1.0;
        tmp2177 += 1.0;
      }
      else if(tmp2178 < -0.5)
      {
        tmp2178 += 1.0;
        tmp2177 -= 1.0;
      }
      if(fabs(tmp2178) < 1e-10 && ((unsigned long)tmp2177 & 1))
      {
        tmp2174 = -pow(-tmp2172, tmp2175)*pow(tmp2172, tmp2176);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2172, tmp2173);
      }
    }
  }
  else
  {
    tmp2174 = pow(tmp2172, tmp2173);
  }
  if(isnan(tmp2174) || isinf(tmp2174))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2172, tmp2173);
  }tmp2179 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2174,"(r_init[283] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2179 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[283] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2179);
    }
  }
  (data->simulationInfo->realParameter[787] /* omega_c[283] PARAM */) = sqrt(tmp2179);
  TRACE_POP
}

/*
equation index: 13437
type: SIMPLE_ASSIGN
r_init[282] = r_min + 282.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13437};
  (data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (282.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13438
type: SIMPLE_ASSIGN
omega_c[282] = sqrt(G * Md / (r_init[282] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13438(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13438};
  modelica_real tmp2180;
  modelica_real tmp2181;
  modelica_real tmp2182;
  modelica_real tmp2183;
  modelica_real tmp2184;
  modelica_real tmp2185;
  modelica_real tmp2186;
  modelica_real tmp2187;
  modelica_real tmp2188;
  modelica_real tmp2189;
  tmp2180 = (data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */);
  tmp2181 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2182 = (tmp2180 * tmp2180) + (tmp2181 * tmp2181);
  tmp2183 = 1.5;
  if(tmp2182 < 0.0 && tmp2183 != 0.0)
  {
    tmp2185 = modf(tmp2183, &tmp2186);
    
    if(tmp2185 > 0.5)
    {
      tmp2185 -= 1.0;
      tmp2186 += 1.0;
    }
    else if(tmp2185 < -0.5)
    {
      tmp2185 += 1.0;
      tmp2186 -= 1.0;
    }
    
    if(fabs(tmp2185) < 1e-10)
      tmp2184 = pow(tmp2182, tmp2186);
    else
    {
      tmp2188 = modf(1.0/tmp2183, &tmp2187);
      if(tmp2188 > 0.5)
      {
        tmp2188 -= 1.0;
        tmp2187 += 1.0;
      }
      else if(tmp2188 < -0.5)
      {
        tmp2188 += 1.0;
        tmp2187 -= 1.0;
      }
      if(fabs(tmp2188) < 1e-10 && ((unsigned long)tmp2187 & 1))
      {
        tmp2184 = -pow(-tmp2182, tmp2185)*pow(tmp2182, tmp2186);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2182, tmp2183);
      }
    }
  }
  else
  {
    tmp2184 = pow(tmp2182, tmp2183);
  }
  if(isnan(tmp2184) || isinf(tmp2184))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2182, tmp2183);
  }tmp2189 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2184,"(r_init[282] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2189 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[282] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2189);
    }
  }
  (data->simulationInfo->realParameter[786] /* omega_c[282] PARAM */) = sqrt(tmp2189);
  TRACE_POP
}

/*
equation index: 13439
type: SIMPLE_ASSIGN
r_init[281] = r_min + 281.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13439};
  (data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (281.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13440
type: SIMPLE_ASSIGN
omega_c[281] = sqrt(G * Md / (r_init[281] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13440(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13440};
  modelica_real tmp2190;
  modelica_real tmp2191;
  modelica_real tmp2192;
  modelica_real tmp2193;
  modelica_real tmp2194;
  modelica_real tmp2195;
  modelica_real tmp2196;
  modelica_real tmp2197;
  modelica_real tmp2198;
  modelica_real tmp2199;
  tmp2190 = (data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */);
  tmp2191 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2192 = (tmp2190 * tmp2190) + (tmp2191 * tmp2191);
  tmp2193 = 1.5;
  if(tmp2192 < 0.0 && tmp2193 != 0.0)
  {
    tmp2195 = modf(tmp2193, &tmp2196);
    
    if(tmp2195 > 0.5)
    {
      tmp2195 -= 1.0;
      tmp2196 += 1.0;
    }
    else if(tmp2195 < -0.5)
    {
      tmp2195 += 1.0;
      tmp2196 -= 1.0;
    }
    
    if(fabs(tmp2195) < 1e-10)
      tmp2194 = pow(tmp2192, tmp2196);
    else
    {
      tmp2198 = modf(1.0/tmp2193, &tmp2197);
      if(tmp2198 > 0.5)
      {
        tmp2198 -= 1.0;
        tmp2197 += 1.0;
      }
      else if(tmp2198 < -0.5)
      {
        tmp2198 += 1.0;
        tmp2197 -= 1.0;
      }
      if(fabs(tmp2198) < 1e-10 && ((unsigned long)tmp2197 & 1))
      {
        tmp2194 = -pow(-tmp2192, tmp2195)*pow(tmp2192, tmp2196);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2192, tmp2193);
      }
    }
  }
  else
  {
    tmp2194 = pow(tmp2192, tmp2193);
  }
  if(isnan(tmp2194) || isinf(tmp2194))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2192, tmp2193);
  }tmp2199 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2194,"(r_init[281] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2199 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[281] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2199);
    }
  }
  (data->simulationInfo->realParameter[785] /* omega_c[281] PARAM */) = sqrt(tmp2199);
  TRACE_POP
}

/*
equation index: 13441
type: SIMPLE_ASSIGN
r_init[280] = r_min + 280.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13441};
  (data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (280.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13442
type: SIMPLE_ASSIGN
omega_c[280] = sqrt(G * Md / (r_init[280] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13442(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13442};
  modelica_real tmp2200;
  modelica_real tmp2201;
  modelica_real tmp2202;
  modelica_real tmp2203;
  modelica_real tmp2204;
  modelica_real tmp2205;
  modelica_real tmp2206;
  modelica_real tmp2207;
  modelica_real tmp2208;
  modelica_real tmp2209;
  tmp2200 = (data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */);
  tmp2201 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2202 = (tmp2200 * tmp2200) + (tmp2201 * tmp2201);
  tmp2203 = 1.5;
  if(tmp2202 < 0.0 && tmp2203 != 0.0)
  {
    tmp2205 = modf(tmp2203, &tmp2206);
    
    if(tmp2205 > 0.5)
    {
      tmp2205 -= 1.0;
      tmp2206 += 1.0;
    }
    else if(tmp2205 < -0.5)
    {
      tmp2205 += 1.0;
      tmp2206 -= 1.0;
    }
    
    if(fabs(tmp2205) < 1e-10)
      tmp2204 = pow(tmp2202, tmp2206);
    else
    {
      tmp2208 = modf(1.0/tmp2203, &tmp2207);
      if(tmp2208 > 0.5)
      {
        tmp2208 -= 1.0;
        tmp2207 += 1.0;
      }
      else if(tmp2208 < -0.5)
      {
        tmp2208 += 1.0;
        tmp2207 -= 1.0;
      }
      if(fabs(tmp2208) < 1e-10 && ((unsigned long)tmp2207 & 1))
      {
        tmp2204 = -pow(-tmp2202, tmp2205)*pow(tmp2202, tmp2206);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2202, tmp2203);
      }
    }
  }
  else
  {
    tmp2204 = pow(tmp2202, tmp2203);
  }
  if(isnan(tmp2204) || isinf(tmp2204))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2202, tmp2203);
  }tmp2209 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2204,"(r_init[280] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2209 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[280] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2209);
    }
  }
  (data->simulationInfo->realParameter[784] /* omega_c[280] PARAM */) = sqrt(tmp2209);
  TRACE_POP
}

/*
equation index: 13443
type: SIMPLE_ASSIGN
r_init[279] = r_min + 279.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13443};
  (data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (279.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13444
type: SIMPLE_ASSIGN
omega_c[279] = sqrt(G * Md / (r_init[279] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13444};
  modelica_real tmp2210;
  modelica_real tmp2211;
  modelica_real tmp2212;
  modelica_real tmp2213;
  modelica_real tmp2214;
  modelica_real tmp2215;
  modelica_real tmp2216;
  modelica_real tmp2217;
  modelica_real tmp2218;
  modelica_real tmp2219;
  tmp2210 = (data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */);
  tmp2211 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2212 = (tmp2210 * tmp2210) + (tmp2211 * tmp2211);
  tmp2213 = 1.5;
  if(tmp2212 < 0.0 && tmp2213 != 0.0)
  {
    tmp2215 = modf(tmp2213, &tmp2216);
    
    if(tmp2215 > 0.5)
    {
      tmp2215 -= 1.0;
      tmp2216 += 1.0;
    }
    else if(tmp2215 < -0.5)
    {
      tmp2215 += 1.0;
      tmp2216 -= 1.0;
    }
    
    if(fabs(tmp2215) < 1e-10)
      tmp2214 = pow(tmp2212, tmp2216);
    else
    {
      tmp2218 = modf(1.0/tmp2213, &tmp2217);
      if(tmp2218 > 0.5)
      {
        tmp2218 -= 1.0;
        tmp2217 += 1.0;
      }
      else if(tmp2218 < -0.5)
      {
        tmp2218 += 1.0;
        tmp2217 -= 1.0;
      }
      if(fabs(tmp2218) < 1e-10 && ((unsigned long)tmp2217 & 1))
      {
        tmp2214 = -pow(-tmp2212, tmp2215)*pow(tmp2212, tmp2216);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2212, tmp2213);
      }
    }
  }
  else
  {
    tmp2214 = pow(tmp2212, tmp2213);
  }
  if(isnan(tmp2214) || isinf(tmp2214))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2212, tmp2213);
  }tmp2219 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2214,"(r_init[279] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2219 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[279] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2219);
    }
  }
  (data->simulationInfo->realParameter[783] /* omega_c[279] PARAM */) = sqrt(tmp2219);
  TRACE_POP
}

/*
equation index: 13445
type: SIMPLE_ASSIGN
r_init[278] = r_min + 278.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13445};
  (data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (278.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13446
type: SIMPLE_ASSIGN
omega_c[278] = sqrt(G * Md / (r_init[278] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13446(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13446};
  modelica_real tmp2220;
  modelica_real tmp2221;
  modelica_real tmp2222;
  modelica_real tmp2223;
  modelica_real tmp2224;
  modelica_real tmp2225;
  modelica_real tmp2226;
  modelica_real tmp2227;
  modelica_real tmp2228;
  modelica_real tmp2229;
  tmp2220 = (data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */);
  tmp2221 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2222 = (tmp2220 * tmp2220) + (tmp2221 * tmp2221);
  tmp2223 = 1.5;
  if(tmp2222 < 0.0 && tmp2223 != 0.0)
  {
    tmp2225 = modf(tmp2223, &tmp2226);
    
    if(tmp2225 > 0.5)
    {
      tmp2225 -= 1.0;
      tmp2226 += 1.0;
    }
    else if(tmp2225 < -0.5)
    {
      tmp2225 += 1.0;
      tmp2226 -= 1.0;
    }
    
    if(fabs(tmp2225) < 1e-10)
      tmp2224 = pow(tmp2222, tmp2226);
    else
    {
      tmp2228 = modf(1.0/tmp2223, &tmp2227);
      if(tmp2228 > 0.5)
      {
        tmp2228 -= 1.0;
        tmp2227 += 1.0;
      }
      else if(tmp2228 < -0.5)
      {
        tmp2228 += 1.0;
        tmp2227 -= 1.0;
      }
      if(fabs(tmp2228) < 1e-10 && ((unsigned long)tmp2227 & 1))
      {
        tmp2224 = -pow(-tmp2222, tmp2225)*pow(tmp2222, tmp2226);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2222, tmp2223);
      }
    }
  }
  else
  {
    tmp2224 = pow(tmp2222, tmp2223);
  }
  if(isnan(tmp2224) || isinf(tmp2224))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2222, tmp2223);
  }tmp2229 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2224,"(r_init[278] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2229 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[278] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2229);
    }
  }
  (data->simulationInfo->realParameter[782] /* omega_c[278] PARAM */) = sqrt(tmp2229);
  TRACE_POP
}

/*
equation index: 13447
type: SIMPLE_ASSIGN
r_init[277] = r_min + 277.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13447};
  (data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (277.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13448
type: SIMPLE_ASSIGN
omega_c[277] = sqrt(G * Md / (r_init[277] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13448(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13448};
  modelica_real tmp2230;
  modelica_real tmp2231;
  modelica_real tmp2232;
  modelica_real tmp2233;
  modelica_real tmp2234;
  modelica_real tmp2235;
  modelica_real tmp2236;
  modelica_real tmp2237;
  modelica_real tmp2238;
  modelica_real tmp2239;
  tmp2230 = (data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */);
  tmp2231 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2232 = (tmp2230 * tmp2230) + (tmp2231 * tmp2231);
  tmp2233 = 1.5;
  if(tmp2232 < 0.0 && tmp2233 != 0.0)
  {
    tmp2235 = modf(tmp2233, &tmp2236);
    
    if(tmp2235 > 0.5)
    {
      tmp2235 -= 1.0;
      tmp2236 += 1.0;
    }
    else if(tmp2235 < -0.5)
    {
      tmp2235 += 1.0;
      tmp2236 -= 1.0;
    }
    
    if(fabs(tmp2235) < 1e-10)
      tmp2234 = pow(tmp2232, tmp2236);
    else
    {
      tmp2238 = modf(1.0/tmp2233, &tmp2237);
      if(tmp2238 > 0.5)
      {
        tmp2238 -= 1.0;
        tmp2237 += 1.0;
      }
      else if(tmp2238 < -0.5)
      {
        tmp2238 += 1.0;
        tmp2237 -= 1.0;
      }
      if(fabs(tmp2238) < 1e-10 && ((unsigned long)tmp2237 & 1))
      {
        tmp2234 = -pow(-tmp2232, tmp2235)*pow(tmp2232, tmp2236);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2232, tmp2233);
      }
    }
  }
  else
  {
    tmp2234 = pow(tmp2232, tmp2233);
  }
  if(isnan(tmp2234) || isinf(tmp2234))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2232, tmp2233);
  }tmp2239 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2234,"(r_init[277] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2239 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[277] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2239);
    }
  }
  (data->simulationInfo->realParameter[781] /* omega_c[277] PARAM */) = sqrt(tmp2239);
  TRACE_POP
}

/*
equation index: 13449
type: SIMPLE_ASSIGN
r_init[276] = r_min + 276.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13449};
  (data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (276.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13450
type: SIMPLE_ASSIGN
omega_c[276] = sqrt(G * Md / (r_init[276] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13450(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13450};
  modelica_real tmp2240;
  modelica_real tmp2241;
  modelica_real tmp2242;
  modelica_real tmp2243;
  modelica_real tmp2244;
  modelica_real tmp2245;
  modelica_real tmp2246;
  modelica_real tmp2247;
  modelica_real tmp2248;
  modelica_real tmp2249;
  tmp2240 = (data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */);
  tmp2241 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2242 = (tmp2240 * tmp2240) + (tmp2241 * tmp2241);
  tmp2243 = 1.5;
  if(tmp2242 < 0.0 && tmp2243 != 0.0)
  {
    tmp2245 = modf(tmp2243, &tmp2246);
    
    if(tmp2245 > 0.5)
    {
      tmp2245 -= 1.0;
      tmp2246 += 1.0;
    }
    else if(tmp2245 < -0.5)
    {
      tmp2245 += 1.0;
      tmp2246 -= 1.0;
    }
    
    if(fabs(tmp2245) < 1e-10)
      tmp2244 = pow(tmp2242, tmp2246);
    else
    {
      tmp2248 = modf(1.0/tmp2243, &tmp2247);
      if(tmp2248 > 0.5)
      {
        tmp2248 -= 1.0;
        tmp2247 += 1.0;
      }
      else if(tmp2248 < -0.5)
      {
        tmp2248 += 1.0;
        tmp2247 -= 1.0;
      }
      if(fabs(tmp2248) < 1e-10 && ((unsigned long)tmp2247 & 1))
      {
        tmp2244 = -pow(-tmp2242, tmp2245)*pow(tmp2242, tmp2246);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2242, tmp2243);
      }
    }
  }
  else
  {
    tmp2244 = pow(tmp2242, tmp2243);
  }
  if(isnan(tmp2244) || isinf(tmp2244))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2242, tmp2243);
  }tmp2249 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2244,"(r_init[276] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2249 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[276] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2249);
    }
  }
  (data->simulationInfo->realParameter[780] /* omega_c[276] PARAM */) = sqrt(tmp2249);
  TRACE_POP
}

/*
equation index: 13451
type: SIMPLE_ASSIGN
r_init[275] = r_min + 275.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13451};
  (data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (275.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13452
type: SIMPLE_ASSIGN
omega_c[275] = sqrt(G * Md / (r_init[275] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13452};
  modelica_real tmp2250;
  modelica_real tmp2251;
  modelica_real tmp2252;
  modelica_real tmp2253;
  modelica_real tmp2254;
  modelica_real tmp2255;
  modelica_real tmp2256;
  modelica_real tmp2257;
  modelica_real tmp2258;
  modelica_real tmp2259;
  tmp2250 = (data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */);
  tmp2251 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2252 = (tmp2250 * tmp2250) + (tmp2251 * tmp2251);
  tmp2253 = 1.5;
  if(tmp2252 < 0.0 && tmp2253 != 0.0)
  {
    tmp2255 = modf(tmp2253, &tmp2256);
    
    if(tmp2255 > 0.5)
    {
      tmp2255 -= 1.0;
      tmp2256 += 1.0;
    }
    else if(tmp2255 < -0.5)
    {
      tmp2255 += 1.0;
      tmp2256 -= 1.0;
    }
    
    if(fabs(tmp2255) < 1e-10)
      tmp2254 = pow(tmp2252, tmp2256);
    else
    {
      tmp2258 = modf(1.0/tmp2253, &tmp2257);
      if(tmp2258 > 0.5)
      {
        tmp2258 -= 1.0;
        tmp2257 += 1.0;
      }
      else if(tmp2258 < -0.5)
      {
        tmp2258 += 1.0;
        tmp2257 -= 1.0;
      }
      if(fabs(tmp2258) < 1e-10 && ((unsigned long)tmp2257 & 1))
      {
        tmp2254 = -pow(-tmp2252, tmp2255)*pow(tmp2252, tmp2256);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2252, tmp2253);
      }
    }
  }
  else
  {
    tmp2254 = pow(tmp2252, tmp2253);
  }
  if(isnan(tmp2254) || isinf(tmp2254))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2252, tmp2253);
  }tmp2259 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2254,"(r_init[275] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2259 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[275] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2259);
    }
  }
  (data->simulationInfo->realParameter[779] /* omega_c[275] PARAM */) = sqrt(tmp2259);
  TRACE_POP
}

/*
equation index: 13453
type: SIMPLE_ASSIGN
r_init[274] = r_min + 274.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13453};
  (data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (274.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13454
type: SIMPLE_ASSIGN
omega_c[274] = sqrt(G * Md / (r_init[274] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13454(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13454};
  modelica_real tmp2260;
  modelica_real tmp2261;
  modelica_real tmp2262;
  modelica_real tmp2263;
  modelica_real tmp2264;
  modelica_real tmp2265;
  modelica_real tmp2266;
  modelica_real tmp2267;
  modelica_real tmp2268;
  modelica_real tmp2269;
  tmp2260 = (data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */);
  tmp2261 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2262 = (tmp2260 * tmp2260) + (tmp2261 * tmp2261);
  tmp2263 = 1.5;
  if(tmp2262 < 0.0 && tmp2263 != 0.0)
  {
    tmp2265 = modf(tmp2263, &tmp2266);
    
    if(tmp2265 > 0.5)
    {
      tmp2265 -= 1.0;
      tmp2266 += 1.0;
    }
    else if(tmp2265 < -0.5)
    {
      tmp2265 += 1.0;
      tmp2266 -= 1.0;
    }
    
    if(fabs(tmp2265) < 1e-10)
      tmp2264 = pow(tmp2262, tmp2266);
    else
    {
      tmp2268 = modf(1.0/tmp2263, &tmp2267);
      if(tmp2268 > 0.5)
      {
        tmp2268 -= 1.0;
        tmp2267 += 1.0;
      }
      else if(tmp2268 < -0.5)
      {
        tmp2268 += 1.0;
        tmp2267 -= 1.0;
      }
      if(fabs(tmp2268) < 1e-10 && ((unsigned long)tmp2267 & 1))
      {
        tmp2264 = -pow(-tmp2262, tmp2265)*pow(tmp2262, tmp2266);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2262, tmp2263);
      }
    }
  }
  else
  {
    tmp2264 = pow(tmp2262, tmp2263);
  }
  if(isnan(tmp2264) || isinf(tmp2264))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2262, tmp2263);
  }tmp2269 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2264,"(r_init[274] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2269 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[274] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2269);
    }
  }
  (data->simulationInfo->realParameter[778] /* omega_c[274] PARAM */) = sqrt(tmp2269);
  TRACE_POP
}

/*
equation index: 13455
type: SIMPLE_ASSIGN
r_init[273] = r_min + 273.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13455};
  (data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (273.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13456
type: SIMPLE_ASSIGN
omega_c[273] = sqrt(G * Md / (r_init[273] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13456(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13456};
  modelica_real tmp2270;
  modelica_real tmp2271;
  modelica_real tmp2272;
  modelica_real tmp2273;
  modelica_real tmp2274;
  modelica_real tmp2275;
  modelica_real tmp2276;
  modelica_real tmp2277;
  modelica_real tmp2278;
  modelica_real tmp2279;
  tmp2270 = (data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */);
  tmp2271 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2272 = (tmp2270 * tmp2270) + (tmp2271 * tmp2271);
  tmp2273 = 1.5;
  if(tmp2272 < 0.0 && tmp2273 != 0.0)
  {
    tmp2275 = modf(tmp2273, &tmp2276);
    
    if(tmp2275 > 0.5)
    {
      tmp2275 -= 1.0;
      tmp2276 += 1.0;
    }
    else if(tmp2275 < -0.5)
    {
      tmp2275 += 1.0;
      tmp2276 -= 1.0;
    }
    
    if(fabs(tmp2275) < 1e-10)
      tmp2274 = pow(tmp2272, tmp2276);
    else
    {
      tmp2278 = modf(1.0/tmp2273, &tmp2277);
      if(tmp2278 > 0.5)
      {
        tmp2278 -= 1.0;
        tmp2277 += 1.0;
      }
      else if(tmp2278 < -0.5)
      {
        tmp2278 += 1.0;
        tmp2277 -= 1.0;
      }
      if(fabs(tmp2278) < 1e-10 && ((unsigned long)tmp2277 & 1))
      {
        tmp2274 = -pow(-tmp2272, tmp2275)*pow(tmp2272, tmp2276);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2272, tmp2273);
      }
    }
  }
  else
  {
    tmp2274 = pow(tmp2272, tmp2273);
  }
  if(isnan(tmp2274) || isinf(tmp2274))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2272, tmp2273);
  }tmp2279 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2274,"(r_init[273] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2279 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[273] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2279);
    }
  }
  (data->simulationInfo->realParameter[777] /* omega_c[273] PARAM */) = sqrt(tmp2279);
  TRACE_POP
}

/*
equation index: 13457
type: SIMPLE_ASSIGN
r_init[272] = r_min + 272.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13457};
  (data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (272.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13458
type: SIMPLE_ASSIGN
omega_c[272] = sqrt(G * Md / (r_init[272] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13458(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13458};
  modelica_real tmp2280;
  modelica_real tmp2281;
  modelica_real tmp2282;
  modelica_real tmp2283;
  modelica_real tmp2284;
  modelica_real tmp2285;
  modelica_real tmp2286;
  modelica_real tmp2287;
  modelica_real tmp2288;
  modelica_real tmp2289;
  tmp2280 = (data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */);
  tmp2281 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2282 = (tmp2280 * tmp2280) + (tmp2281 * tmp2281);
  tmp2283 = 1.5;
  if(tmp2282 < 0.0 && tmp2283 != 0.0)
  {
    tmp2285 = modf(tmp2283, &tmp2286);
    
    if(tmp2285 > 0.5)
    {
      tmp2285 -= 1.0;
      tmp2286 += 1.0;
    }
    else if(tmp2285 < -0.5)
    {
      tmp2285 += 1.0;
      tmp2286 -= 1.0;
    }
    
    if(fabs(tmp2285) < 1e-10)
      tmp2284 = pow(tmp2282, tmp2286);
    else
    {
      tmp2288 = modf(1.0/tmp2283, &tmp2287);
      if(tmp2288 > 0.5)
      {
        tmp2288 -= 1.0;
        tmp2287 += 1.0;
      }
      else if(tmp2288 < -0.5)
      {
        tmp2288 += 1.0;
        tmp2287 -= 1.0;
      }
      if(fabs(tmp2288) < 1e-10 && ((unsigned long)tmp2287 & 1))
      {
        tmp2284 = -pow(-tmp2282, tmp2285)*pow(tmp2282, tmp2286);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2282, tmp2283);
      }
    }
  }
  else
  {
    tmp2284 = pow(tmp2282, tmp2283);
  }
  if(isnan(tmp2284) || isinf(tmp2284))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2282, tmp2283);
  }tmp2289 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2284,"(r_init[272] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2289 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[272] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2289);
    }
  }
  (data->simulationInfo->realParameter[776] /* omega_c[272] PARAM */) = sqrt(tmp2289);
  TRACE_POP
}

/*
equation index: 13459
type: SIMPLE_ASSIGN
r_init[271] = r_min + 271.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13459};
  (data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (271.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13460
type: SIMPLE_ASSIGN
omega_c[271] = sqrt(G * Md / (r_init[271] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13460};
  modelica_real tmp2290;
  modelica_real tmp2291;
  modelica_real tmp2292;
  modelica_real tmp2293;
  modelica_real tmp2294;
  modelica_real tmp2295;
  modelica_real tmp2296;
  modelica_real tmp2297;
  modelica_real tmp2298;
  modelica_real tmp2299;
  tmp2290 = (data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */);
  tmp2291 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2292 = (tmp2290 * tmp2290) + (tmp2291 * tmp2291);
  tmp2293 = 1.5;
  if(tmp2292 < 0.0 && tmp2293 != 0.0)
  {
    tmp2295 = modf(tmp2293, &tmp2296);
    
    if(tmp2295 > 0.5)
    {
      tmp2295 -= 1.0;
      tmp2296 += 1.0;
    }
    else if(tmp2295 < -0.5)
    {
      tmp2295 += 1.0;
      tmp2296 -= 1.0;
    }
    
    if(fabs(tmp2295) < 1e-10)
      tmp2294 = pow(tmp2292, tmp2296);
    else
    {
      tmp2298 = modf(1.0/tmp2293, &tmp2297);
      if(tmp2298 > 0.5)
      {
        tmp2298 -= 1.0;
        tmp2297 += 1.0;
      }
      else if(tmp2298 < -0.5)
      {
        tmp2298 += 1.0;
        tmp2297 -= 1.0;
      }
      if(fabs(tmp2298) < 1e-10 && ((unsigned long)tmp2297 & 1))
      {
        tmp2294 = -pow(-tmp2292, tmp2295)*pow(tmp2292, tmp2296);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2292, tmp2293);
      }
    }
  }
  else
  {
    tmp2294 = pow(tmp2292, tmp2293);
  }
  if(isnan(tmp2294) || isinf(tmp2294))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2292, tmp2293);
  }tmp2299 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2294,"(r_init[271] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2299 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[271] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2299);
    }
  }
  (data->simulationInfo->realParameter[775] /* omega_c[271] PARAM */) = sqrt(tmp2299);
  TRACE_POP
}

/*
equation index: 13461
type: SIMPLE_ASSIGN
r_init[270] = r_min + 270.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13461};
  (data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (270.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13462
type: SIMPLE_ASSIGN
omega_c[270] = sqrt(G * Md / (r_init[270] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13462(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13462};
  modelica_real tmp2300;
  modelica_real tmp2301;
  modelica_real tmp2302;
  modelica_real tmp2303;
  modelica_real tmp2304;
  modelica_real tmp2305;
  modelica_real tmp2306;
  modelica_real tmp2307;
  modelica_real tmp2308;
  modelica_real tmp2309;
  tmp2300 = (data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */);
  tmp2301 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2302 = (tmp2300 * tmp2300) + (tmp2301 * tmp2301);
  tmp2303 = 1.5;
  if(tmp2302 < 0.0 && tmp2303 != 0.0)
  {
    tmp2305 = modf(tmp2303, &tmp2306);
    
    if(tmp2305 > 0.5)
    {
      tmp2305 -= 1.0;
      tmp2306 += 1.0;
    }
    else if(tmp2305 < -0.5)
    {
      tmp2305 += 1.0;
      tmp2306 -= 1.0;
    }
    
    if(fabs(tmp2305) < 1e-10)
      tmp2304 = pow(tmp2302, tmp2306);
    else
    {
      tmp2308 = modf(1.0/tmp2303, &tmp2307);
      if(tmp2308 > 0.5)
      {
        tmp2308 -= 1.0;
        tmp2307 += 1.0;
      }
      else if(tmp2308 < -0.5)
      {
        tmp2308 += 1.0;
        tmp2307 -= 1.0;
      }
      if(fabs(tmp2308) < 1e-10 && ((unsigned long)tmp2307 & 1))
      {
        tmp2304 = -pow(-tmp2302, tmp2305)*pow(tmp2302, tmp2306);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2302, tmp2303);
      }
    }
  }
  else
  {
    tmp2304 = pow(tmp2302, tmp2303);
  }
  if(isnan(tmp2304) || isinf(tmp2304))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2302, tmp2303);
  }tmp2309 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2304,"(r_init[270] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2309 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[270] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2309);
    }
  }
  (data->simulationInfo->realParameter[774] /* omega_c[270] PARAM */) = sqrt(tmp2309);
  TRACE_POP
}

/*
equation index: 13463
type: SIMPLE_ASSIGN
r_init[269] = r_min + 269.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13463};
  (data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (269.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13464
type: SIMPLE_ASSIGN
omega_c[269] = sqrt(G * Md / (r_init[269] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13464(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13464};
  modelica_real tmp2310;
  modelica_real tmp2311;
  modelica_real tmp2312;
  modelica_real tmp2313;
  modelica_real tmp2314;
  modelica_real tmp2315;
  modelica_real tmp2316;
  modelica_real tmp2317;
  modelica_real tmp2318;
  modelica_real tmp2319;
  tmp2310 = (data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */);
  tmp2311 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2312 = (tmp2310 * tmp2310) + (tmp2311 * tmp2311);
  tmp2313 = 1.5;
  if(tmp2312 < 0.0 && tmp2313 != 0.0)
  {
    tmp2315 = modf(tmp2313, &tmp2316);
    
    if(tmp2315 > 0.5)
    {
      tmp2315 -= 1.0;
      tmp2316 += 1.0;
    }
    else if(tmp2315 < -0.5)
    {
      tmp2315 += 1.0;
      tmp2316 -= 1.0;
    }
    
    if(fabs(tmp2315) < 1e-10)
      tmp2314 = pow(tmp2312, tmp2316);
    else
    {
      tmp2318 = modf(1.0/tmp2313, &tmp2317);
      if(tmp2318 > 0.5)
      {
        tmp2318 -= 1.0;
        tmp2317 += 1.0;
      }
      else if(tmp2318 < -0.5)
      {
        tmp2318 += 1.0;
        tmp2317 -= 1.0;
      }
      if(fabs(tmp2318) < 1e-10 && ((unsigned long)tmp2317 & 1))
      {
        tmp2314 = -pow(-tmp2312, tmp2315)*pow(tmp2312, tmp2316);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2312, tmp2313);
      }
    }
  }
  else
  {
    tmp2314 = pow(tmp2312, tmp2313);
  }
  if(isnan(tmp2314) || isinf(tmp2314))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2312, tmp2313);
  }tmp2319 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2314,"(r_init[269] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2319 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[269] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2319);
    }
  }
  (data->simulationInfo->realParameter[773] /* omega_c[269] PARAM */) = sqrt(tmp2319);
  TRACE_POP
}

/*
equation index: 13465
type: SIMPLE_ASSIGN
r_init[268] = r_min + 268.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13465};
  (data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (268.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13466
type: SIMPLE_ASSIGN
omega_c[268] = sqrt(G * Md / (r_init[268] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13466(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13466};
  modelica_real tmp2320;
  modelica_real tmp2321;
  modelica_real tmp2322;
  modelica_real tmp2323;
  modelica_real tmp2324;
  modelica_real tmp2325;
  modelica_real tmp2326;
  modelica_real tmp2327;
  modelica_real tmp2328;
  modelica_real tmp2329;
  tmp2320 = (data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */);
  tmp2321 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2322 = (tmp2320 * tmp2320) + (tmp2321 * tmp2321);
  tmp2323 = 1.5;
  if(tmp2322 < 0.0 && tmp2323 != 0.0)
  {
    tmp2325 = modf(tmp2323, &tmp2326);
    
    if(tmp2325 > 0.5)
    {
      tmp2325 -= 1.0;
      tmp2326 += 1.0;
    }
    else if(tmp2325 < -0.5)
    {
      tmp2325 += 1.0;
      tmp2326 -= 1.0;
    }
    
    if(fabs(tmp2325) < 1e-10)
      tmp2324 = pow(tmp2322, tmp2326);
    else
    {
      tmp2328 = modf(1.0/tmp2323, &tmp2327);
      if(tmp2328 > 0.5)
      {
        tmp2328 -= 1.0;
        tmp2327 += 1.0;
      }
      else if(tmp2328 < -0.5)
      {
        tmp2328 += 1.0;
        tmp2327 -= 1.0;
      }
      if(fabs(tmp2328) < 1e-10 && ((unsigned long)tmp2327 & 1))
      {
        tmp2324 = -pow(-tmp2322, tmp2325)*pow(tmp2322, tmp2326);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2322, tmp2323);
      }
    }
  }
  else
  {
    tmp2324 = pow(tmp2322, tmp2323);
  }
  if(isnan(tmp2324) || isinf(tmp2324))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2322, tmp2323);
  }tmp2329 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2324,"(r_init[268] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2329 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[268] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2329);
    }
  }
  (data->simulationInfo->realParameter[772] /* omega_c[268] PARAM */) = sqrt(tmp2329);
  TRACE_POP
}

/*
equation index: 13467
type: SIMPLE_ASSIGN
r_init[267] = r_min + 267.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13467};
  (data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (267.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13468
type: SIMPLE_ASSIGN
omega_c[267] = sqrt(G * Md / (r_init[267] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13468};
  modelica_real tmp2330;
  modelica_real tmp2331;
  modelica_real tmp2332;
  modelica_real tmp2333;
  modelica_real tmp2334;
  modelica_real tmp2335;
  modelica_real tmp2336;
  modelica_real tmp2337;
  modelica_real tmp2338;
  modelica_real tmp2339;
  tmp2330 = (data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */);
  tmp2331 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2332 = (tmp2330 * tmp2330) + (tmp2331 * tmp2331);
  tmp2333 = 1.5;
  if(tmp2332 < 0.0 && tmp2333 != 0.0)
  {
    tmp2335 = modf(tmp2333, &tmp2336);
    
    if(tmp2335 > 0.5)
    {
      tmp2335 -= 1.0;
      tmp2336 += 1.0;
    }
    else if(tmp2335 < -0.5)
    {
      tmp2335 += 1.0;
      tmp2336 -= 1.0;
    }
    
    if(fabs(tmp2335) < 1e-10)
      tmp2334 = pow(tmp2332, tmp2336);
    else
    {
      tmp2338 = modf(1.0/tmp2333, &tmp2337);
      if(tmp2338 > 0.5)
      {
        tmp2338 -= 1.0;
        tmp2337 += 1.0;
      }
      else if(tmp2338 < -0.5)
      {
        tmp2338 += 1.0;
        tmp2337 -= 1.0;
      }
      if(fabs(tmp2338) < 1e-10 && ((unsigned long)tmp2337 & 1))
      {
        tmp2334 = -pow(-tmp2332, tmp2335)*pow(tmp2332, tmp2336);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2332, tmp2333);
      }
    }
  }
  else
  {
    tmp2334 = pow(tmp2332, tmp2333);
  }
  if(isnan(tmp2334) || isinf(tmp2334))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2332, tmp2333);
  }tmp2339 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2334,"(r_init[267] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2339 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[267] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2339);
    }
  }
  (data->simulationInfo->realParameter[771] /* omega_c[267] PARAM */) = sqrt(tmp2339);
  TRACE_POP
}

/*
equation index: 13469
type: SIMPLE_ASSIGN
r_init[266] = r_min + 266.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13469};
  (data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (266.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13470
type: SIMPLE_ASSIGN
omega_c[266] = sqrt(G * Md / (r_init[266] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13470(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13470};
  modelica_real tmp2340;
  modelica_real tmp2341;
  modelica_real tmp2342;
  modelica_real tmp2343;
  modelica_real tmp2344;
  modelica_real tmp2345;
  modelica_real tmp2346;
  modelica_real tmp2347;
  modelica_real tmp2348;
  modelica_real tmp2349;
  tmp2340 = (data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */);
  tmp2341 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2342 = (tmp2340 * tmp2340) + (tmp2341 * tmp2341);
  tmp2343 = 1.5;
  if(tmp2342 < 0.0 && tmp2343 != 0.0)
  {
    tmp2345 = modf(tmp2343, &tmp2346);
    
    if(tmp2345 > 0.5)
    {
      tmp2345 -= 1.0;
      tmp2346 += 1.0;
    }
    else if(tmp2345 < -0.5)
    {
      tmp2345 += 1.0;
      tmp2346 -= 1.0;
    }
    
    if(fabs(tmp2345) < 1e-10)
      tmp2344 = pow(tmp2342, tmp2346);
    else
    {
      tmp2348 = modf(1.0/tmp2343, &tmp2347);
      if(tmp2348 > 0.5)
      {
        tmp2348 -= 1.0;
        tmp2347 += 1.0;
      }
      else if(tmp2348 < -0.5)
      {
        tmp2348 += 1.0;
        tmp2347 -= 1.0;
      }
      if(fabs(tmp2348) < 1e-10 && ((unsigned long)tmp2347 & 1))
      {
        tmp2344 = -pow(-tmp2342, tmp2345)*pow(tmp2342, tmp2346);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2342, tmp2343);
      }
    }
  }
  else
  {
    tmp2344 = pow(tmp2342, tmp2343);
  }
  if(isnan(tmp2344) || isinf(tmp2344))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2342, tmp2343);
  }tmp2349 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2344,"(r_init[266] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2349 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[266] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2349);
    }
  }
  (data->simulationInfo->realParameter[770] /* omega_c[266] PARAM */) = sqrt(tmp2349);
  TRACE_POP
}

/*
equation index: 13471
type: SIMPLE_ASSIGN
r_init[265] = r_min + 265.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13471};
  (data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (265.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13472
type: SIMPLE_ASSIGN
omega_c[265] = sqrt(G * Md / (r_init[265] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13472(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13472};
  modelica_real tmp2350;
  modelica_real tmp2351;
  modelica_real tmp2352;
  modelica_real tmp2353;
  modelica_real tmp2354;
  modelica_real tmp2355;
  modelica_real tmp2356;
  modelica_real tmp2357;
  modelica_real tmp2358;
  modelica_real tmp2359;
  tmp2350 = (data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */);
  tmp2351 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2352 = (tmp2350 * tmp2350) + (tmp2351 * tmp2351);
  tmp2353 = 1.5;
  if(tmp2352 < 0.0 && tmp2353 != 0.0)
  {
    tmp2355 = modf(tmp2353, &tmp2356);
    
    if(tmp2355 > 0.5)
    {
      tmp2355 -= 1.0;
      tmp2356 += 1.0;
    }
    else if(tmp2355 < -0.5)
    {
      tmp2355 += 1.0;
      tmp2356 -= 1.0;
    }
    
    if(fabs(tmp2355) < 1e-10)
      tmp2354 = pow(tmp2352, tmp2356);
    else
    {
      tmp2358 = modf(1.0/tmp2353, &tmp2357);
      if(tmp2358 > 0.5)
      {
        tmp2358 -= 1.0;
        tmp2357 += 1.0;
      }
      else if(tmp2358 < -0.5)
      {
        tmp2358 += 1.0;
        tmp2357 -= 1.0;
      }
      if(fabs(tmp2358) < 1e-10 && ((unsigned long)tmp2357 & 1))
      {
        tmp2354 = -pow(-tmp2352, tmp2355)*pow(tmp2352, tmp2356);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2352, tmp2353);
      }
    }
  }
  else
  {
    tmp2354 = pow(tmp2352, tmp2353);
  }
  if(isnan(tmp2354) || isinf(tmp2354))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2352, tmp2353);
  }tmp2359 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2354,"(r_init[265] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2359 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[265] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2359);
    }
  }
  (data->simulationInfo->realParameter[769] /* omega_c[265] PARAM */) = sqrt(tmp2359);
  TRACE_POP
}

/*
equation index: 13473
type: SIMPLE_ASSIGN
r_init[264] = r_min + 264.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13473};
  (data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (264.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13474
type: SIMPLE_ASSIGN
omega_c[264] = sqrt(G * Md / (r_init[264] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13474(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13474};
  modelica_real tmp2360;
  modelica_real tmp2361;
  modelica_real tmp2362;
  modelica_real tmp2363;
  modelica_real tmp2364;
  modelica_real tmp2365;
  modelica_real tmp2366;
  modelica_real tmp2367;
  modelica_real tmp2368;
  modelica_real tmp2369;
  tmp2360 = (data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */);
  tmp2361 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2362 = (tmp2360 * tmp2360) + (tmp2361 * tmp2361);
  tmp2363 = 1.5;
  if(tmp2362 < 0.0 && tmp2363 != 0.0)
  {
    tmp2365 = modf(tmp2363, &tmp2366);
    
    if(tmp2365 > 0.5)
    {
      tmp2365 -= 1.0;
      tmp2366 += 1.0;
    }
    else if(tmp2365 < -0.5)
    {
      tmp2365 += 1.0;
      tmp2366 -= 1.0;
    }
    
    if(fabs(tmp2365) < 1e-10)
      tmp2364 = pow(tmp2362, tmp2366);
    else
    {
      tmp2368 = modf(1.0/tmp2363, &tmp2367);
      if(tmp2368 > 0.5)
      {
        tmp2368 -= 1.0;
        tmp2367 += 1.0;
      }
      else if(tmp2368 < -0.5)
      {
        tmp2368 += 1.0;
        tmp2367 -= 1.0;
      }
      if(fabs(tmp2368) < 1e-10 && ((unsigned long)tmp2367 & 1))
      {
        tmp2364 = -pow(-tmp2362, tmp2365)*pow(tmp2362, tmp2366);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2362, tmp2363);
      }
    }
  }
  else
  {
    tmp2364 = pow(tmp2362, tmp2363);
  }
  if(isnan(tmp2364) || isinf(tmp2364))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2362, tmp2363);
  }tmp2369 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2364,"(r_init[264] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2369 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[264] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2369);
    }
  }
  (data->simulationInfo->realParameter[768] /* omega_c[264] PARAM */) = sqrt(tmp2369);
  TRACE_POP
}

/*
equation index: 13475
type: SIMPLE_ASSIGN
r_init[263] = r_min + 263.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13475};
  (data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (263.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13476
type: SIMPLE_ASSIGN
omega_c[263] = sqrt(G * Md / (r_init[263] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13476};
  modelica_real tmp2370;
  modelica_real tmp2371;
  modelica_real tmp2372;
  modelica_real tmp2373;
  modelica_real tmp2374;
  modelica_real tmp2375;
  modelica_real tmp2376;
  modelica_real tmp2377;
  modelica_real tmp2378;
  modelica_real tmp2379;
  tmp2370 = (data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */);
  tmp2371 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2372 = (tmp2370 * tmp2370) + (tmp2371 * tmp2371);
  tmp2373 = 1.5;
  if(tmp2372 < 0.0 && tmp2373 != 0.0)
  {
    tmp2375 = modf(tmp2373, &tmp2376);
    
    if(tmp2375 > 0.5)
    {
      tmp2375 -= 1.0;
      tmp2376 += 1.0;
    }
    else if(tmp2375 < -0.5)
    {
      tmp2375 += 1.0;
      tmp2376 -= 1.0;
    }
    
    if(fabs(tmp2375) < 1e-10)
      tmp2374 = pow(tmp2372, tmp2376);
    else
    {
      tmp2378 = modf(1.0/tmp2373, &tmp2377);
      if(tmp2378 > 0.5)
      {
        tmp2378 -= 1.0;
        tmp2377 += 1.0;
      }
      else if(tmp2378 < -0.5)
      {
        tmp2378 += 1.0;
        tmp2377 -= 1.0;
      }
      if(fabs(tmp2378) < 1e-10 && ((unsigned long)tmp2377 & 1))
      {
        tmp2374 = -pow(-tmp2372, tmp2375)*pow(tmp2372, tmp2376);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2372, tmp2373);
      }
    }
  }
  else
  {
    tmp2374 = pow(tmp2372, tmp2373);
  }
  if(isnan(tmp2374) || isinf(tmp2374))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2372, tmp2373);
  }tmp2379 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2374,"(r_init[263] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2379 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[263] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2379);
    }
  }
  (data->simulationInfo->realParameter[767] /* omega_c[263] PARAM */) = sqrt(tmp2379);
  TRACE_POP
}

/*
equation index: 13477
type: SIMPLE_ASSIGN
r_init[262] = r_min + 262.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13477};
  (data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (262.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13478
type: SIMPLE_ASSIGN
omega_c[262] = sqrt(G * Md / (r_init[262] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13478(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13478};
  modelica_real tmp2380;
  modelica_real tmp2381;
  modelica_real tmp2382;
  modelica_real tmp2383;
  modelica_real tmp2384;
  modelica_real tmp2385;
  modelica_real tmp2386;
  modelica_real tmp2387;
  modelica_real tmp2388;
  modelica_real tmp2389;
  tmp2380 = (data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */);
  tmp2381 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2382 = (tmp2380 * tmp2380) + (tmp2381 * tmp2381);
  tmp2383 = 1.5;
  if(tmp2382 < 0.0 && tmp2383 != 0.0)
  {
    tmp2385 = modf(tmp2383, &tmp2386);
    
    if(tmp2385 > 0.5)
    {
      tmp2385 -= 1.0;
      tmp2386 += 1.0;
    }
    else if(tmp2385 < -0.5)
    {
      tmp2385 += 1.0;
      tmp2386 -= 1.0;
    }
    
    if(fabs(tmp2385) < 1e-10)
      tmp2384 = pow(tmp2382, tmp2386);
    else
    {
      tmp2388 = modf(1.0/tmp2383, &tmp2387);
      if(tmp2388 > 0.5)
      {
        tmp2388 -= 1.0;
        tmp2387 += 1.0;
      }
      else if(tmp2388 < -0.5)
      {
        tmp2388 += 1.0;
        tmp2387 -= 1.0;
      }
      if(fabs(tmp2388) < 1e-10 && ((unsigned long)tmp2387 & 1))
      {
        tmp2384 = -pow(-tmp2382, tmp2385)*pow(tmp2382, tmp2386);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2382, tmp2383);
      }
    }
  }
  else
  {
    tmp2384 = pow(tmp2382, tmp2383);
  }
  if(isnan(tmp2384) || isinf(tmp2384))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2382, tmp2383);
  }tmp2389 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2384,"(r_init[262] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2389 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[262] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2389);
    }
  }
  (data->simulationInfo->realParameter[766] /* omega_c[262] PARAM */) = sqrt(tmp2389);
  TRACE_POP
}

/*
equation index: 13479
type: SIMPLE_ASSIGN
r_init[261] = r_min + 261.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13479};
  (data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (261.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13480
type: SIMPLE_ASSIGN
omega_c[261] = sqrt(G * Md / (r_init[261] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13480(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13480};
  modelica_real tmp2390;
  modelica_real tmp2391;
  modelica_real tmp2392;
  modelica_real tmp2393;
  modelica_real tmp2394;
  modelica_real tmp2395;
  modelica_real tmp2396;
  modelica_real tmp2397;
  modelica_real tmp2398;
  modelica_real tmp2399;
  tmp2390 = (data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */);
  tmp2391 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2392 = (tmp2390 * tmp2390) + (tmp2391 * tmp2391);
  tmp2393 = 1.5;
  if(tmp2392 < 0.0 && tmp2393 != 0.0)
  {
    tmp2395 = modf(tmp2393, &tmp2396);
    
    if(tmp2395 > 0.5)
    {
      tmp2395 -= 1.0;
      tmp2396 += 1.0;
    }
    else if(tmp2395 < -0.5)
    {
      tmp2395 += 1.0;
      tmp2396 -= 1.0;
    }
    
    if(fabs(tmp2395) < 1e-10)
      tmp2394 = pow(tmp2392, tmp2396);
    else
    {
      tmp2398 = modf(1.0/tmp2393, &tmp2397);
      if(tmp2398 > 0.5)
      {
        tmp2398 -= 1.0;
        tmp2397 += 1.0;
      }
      else if(tmp2398 < -0.5)
      {
        tmp2398 += 1.0;
        tmp2397 -= 1.0;
      }
      if(fabs(tmp2398) < 1e-10 && ((unsigned long)tmp2397 & 1))
      {
        tmp2394 = -pow(-tmp2392, tmp2395)*pow(tmp2392, tmp2396);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2392, tmp2393);
      }
    }
  }
  else
  {
    tmp2394 = pow(tmp2392, tmp2393);
  }
  if(isnan(tmp2394) || isinf(tmp2394))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2392, tmp2393);
  }tmp2399 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2394,"(r_init[261] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2399 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[261] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2399);
    }
  }
  (data->simulationInfo->realParameter[765] /* omega_c[261] PARAM */) = sqrt(tmp2399);
  TRACE_POP
}

/*
equation index: 13481
type: SIMPLE_ASSIGN
r_init[260] = r_min + 260.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13481};
  (data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (260.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13482
type: SIMPLE_ASSIGN
omega_c[260] = sqrt(G * Md / (r_init[260] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13482(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13482};
  modelica_real tmp2400;
  modelica_real tmp2401;
  modelica_real tmp2402;
  modelica_real tmp2403;
  modelica_real tmp2404;
  modelica_real tmp2405;
  modelica_real tmp2406;
  modelica_real tmp2407;
  modelica_real tmp2408;
  modelica_real tmp2409;
  tmp2400 = (data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */);
  tmp2401 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2402 = (tmp2400 * tmp2400) + (tmp2401 * tmp2401);
  tmp2403 = 1.5;
  if(tmp2402 < 0.0 && tmp2403 != 0.0)
  {
    tmp2405 = modf(tmp2403, &tmp2406);
    
    if(tmp2405 > 0.5)
    {
      tmp2405 -= 1.0;
      tmp2406 += 1.0;
    }
    else if(tmp2405 < -0.5)
    {
      tmp2405 += 1.0;
      tmp2406 -= 1.0;
    }
    
    if(fabs(tmp2405) < 1e-10)
      tmp2404 = pow(tmp2402, tmp2406);
    else
    {
      tmp2408 = modf(1.0/tmp2403, &tmp2407);
      if(tmp2408 > 0.5)
      {
        tmp2408 -= 1.0;
        tmp2407 += 1.0;
      }
      else if(tmp2408 < -0.5)
      {
        tmp2408 += 1.0;
        tmp2407 -= 1.0;
      }
      if(fabs(tmp2408) < 1e-10 && ((unsigned long)tmp2407 & 1))
      {
        tmp2404 = -pow(-tmp2402, tmp2405)*pow(tmp2402, tmp2406);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2402, tmp2403);
      }
    }
  }
  else
  {
    tmp2404 = pow(tmp2402, tmp2403);
  }
  if(isnan(tmp2404) || isinf(tmp2404))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2402, tmp2403);
  }tmp2409 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2404,"(r_init[260] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2409 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[260] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2409);
    }
  }
  (data->simulationInfo->realParameter[764] /* omega_c[260] PARAM */) = sqrt(tmp2409);
  TRACE_POP
}

/*
equation index: 13483
type: SIMPLE_ASSIGN
r_init[259] = r_min + 259.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13483};
  (data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (259.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13484
type: SIMPLE_ASSIGN
omega_c[259] = sqrt(G * Md / (r_init[259] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13484};
  modelica_real tmp2410;
  modelica_real tmp2411;
  modelica_real tmp2412;
  modelica_real tmp2413;
  modelica_real tmp2414;
  modelica_real tmp2415;
  modelica_real tmp2416;
  modelica_real tmp2417;
  modelica_real tmp2418;
  modelica_real tmp2419;
  tmp2410 = (data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */);
  tmp2411 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2412 = (tmp2410 * tmp2410) + (tmp2411 * tmp2411);
  tmp2413 = 1.5;
  if(tmp2412 < 0.0 && tmp2413 != 0.0)
  {
    tmp2415 = modf(tmp2413, &tmp2416);
    
    if(tmp2415 > 0.5)
    {
      tmp2415 -= 1.0;
      tmp2416 += 1.0;
    }
    else if(tmp2415 < -0.5)
    {
      tmp2415 += 1.0;
      tmp2416 -= 1.0;
    }
    
    if(fabs(tmp2415) < 1e-10)
      tmp2414 = pow(tmp2412, tmp2416);
    else
    {
      tmp2418 = modf(1.0/tmp2413, &tmp2417);
      if(tmp2418 > 0.5)
      {
        tmp2418 -= 1.0;
        tmp2417 += 1.0;
      }
      else if(tmp2418 < -0.5)
      {
        tmp2418 += 1.0;
        tmp2417 -= 1.0;
      }
      if(fabs(tmp2418) < 1e-10 && ((unsigned long)tmp2417 & 1))
      {
        tmp2414 = -pow(-tmp2412, tmp2415)*pow(tmp2412, tmp2416);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2412, tmp2413);
      }
    }
  }
  else
  {
    tmp2414 = pow(tmp2412, tmp2413);
  }
  if(isnan(tmp2414) || isinf(tmp2414))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2412, tmp2413);
  }tmp2419 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2414,"(r_init[259] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2419 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[259] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2419);
    }
  }
  (data->simulationInfo->realParameter[763] /* omega_c[259] PARAM */) = sqrt(tmp2419);
  TRACE_POP
}

/*
equation index: 13485
type: SIMPLE_ASSIGN
r_init[258] = r_min + 258.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13485};
  (data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (258.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13486
type: SIMPLE_ASSIGN
omega_c[258] = sqrt(G * Md / (r_init[258] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13486(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13486};
  modelica_real tmp2420;
  modelica_real tmp2421;
  modelica_real tmp2422;
  modelica_real tmp2423;
  modelica_real tmp2424;
  modelica_real tmp2425;
  modelica_real tmp2426;
  modelica_real tmp2427;
  modelica_real tmp2428;
  modelica_real tmp2429;
  tmp2420 = (data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */);
  tmp2421 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2422 = (tmp2420 * tmp2420) + (tmp2421 * tmp2421);
  tmp2423 = 1.5;
  if(tmp2422 < 0.0 && tmp2423 != 0.0)
  {
    tmp2425 = modf(tmp2423, &tmp2426);
    
    if(tmp2425 > 0.5)
    {
      tmp2425 -= 1.0;
      tmp2426 += 1.0;
    }
    else if(tmp2425 < -0.5)
    {
      tmp2425 += 1.0;
      tmp2426 -= 1.0;
    }
    
    if(fabs(tmp2425) < 1e-10)
      tmp2424 = pow(tmp2422, tmp2426);
    else
    {
      tmp2428 = modf(1.0/tmp2423, &tmp2427);
      if(tmp2428 > 0.5)
      {
        tmp2428 -= 1.0;
        tmp2427 += 1.0;
      }
      else if(tmp2428 < -0.5)
      {
        tmp2428 += 1.0;
        tmp2427 -= 1.0;
      }
      if(fabs(tmp2428) < 1e-10 && ((unsigned long)tmp2427 & 1))
      {
        tmp2424 = -pow(-tmp2422, tmp2425)*pow(tmp2422, tmp2426);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2422, tmp2423);
      }
    }
  }
  else
  {
    tmp2424 = pow(tmp2422, tmp2423);
  }
  if(isnan(tmp2424) || isinf(tmp2424))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2422, tmp2423);
  }tmp2429 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2424,"(r_init[258] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2429 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[258] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2429);
    }
  }
  (data->simulationInfo->realParameter[762] /* omega_c[258] PARAM */) = sqrt(tmp2429);
  TRACE_POP
}

/*
equation index: 13487
type: SIMPLE_ASSIGN
r_init[257] = r_min + 257.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13487};
  (data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (257.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13488
type: SIMPLE_ASSIGN
omega_c[257] = sqrt(G * Md / (r_init[257] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13488(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13488};
  modelica_real tmp2430;
  modelica_real tmp2431;
  modelica_real tmp2432;
  modelica_real tmp2433;
  modelica_real tmp2434;
  modelica_real tmp2435;
  modelica_real tmp2436;
  modelica_real tmp2437;
  modelica_real tmp2438;
  modelica_real tmp2439;
  tmp2430 = (data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */);
  tmp2431 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2432 = (tmp2430 * tmp2430) + (tmp2431 * tmp2431);
  tmp2433 = 1.5;
  if(tmp2432 < 0.0 && tmp2433 != 0.0)
  {
    tmp2435 = modf(tmp2433, &tmp2436);
    
    if(tmp2435 > 0.5)
    {
      tmp2435 -= 1.0;
      tmp2436 += 1.0;
    }
    else if(tmp2435 < -0.5)
    {
      tmp2435 += 1.0;
      tmp2436 -= 1.0;
    }
    
    if(fabs(tmp2435) < 1e-10)
      tmp2434 = pow(tmp2432, tmp2436);
    else
    {
      tmp2438 = modf(1.0/tmp2433, &tmp2437);
      if(tmp2438 > 0.5)
      {
        tmp2438 -= 1.0;
        tmp2437 += 1.0;
      }
      else if(tmp2438 < -0.5)
      {
        tmp2438 += 1.0;
        tmp2437 -= 1.0;
      }
      if(fabs(tmp2438) < 1e-10 && ((unsigned long)tmp2437 & 1))
      {
        tmp2434 = -pow(-tmp2432, tmp2435)*pow(tmp2432, tmp2436);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2432, tmp2433);
      }
    }
  }
  else
  {
    tmp2434 = pow(tmp2432, tmp2433);
  }
  if(isnan(tmp2434) || isinf(tmp2434))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2432, tmp2433);
  }tmp2439 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2434,"(r_init[257] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2439 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[257] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2439);
    }
  }
  (data->simulationInfo->realParameter[761] /* omega_c[257] PARAM */) = sqrt(tmp2439);
  TRACE_POP
}

/*
equation index: 13489
type: SIMPLE_ASSIGN
r_init[256] = r_min + 256.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13489};
  (data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (256.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13490
type: SIMPLE_ASSIGN
omega_c[256] = sqrt(G * Md / (r_init[256] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13490(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13490};
  modelica_real tmp2440;
  modelica_real tmp2441;
  modelica_real tmp2442;
  modelica_real tmp2443;
  modelica_real tmp2444;
  modelica_real tmp2445;
  modelica_real tmp2446;
  modelica_real tmp2447;
  modelica_real tmp2448;
  modelica_real tmp2449;
  tmp2440 = (data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */);
  tmp2441 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2442 = (tmp2440 * tmp2440) + (tmp2441 * tmp2441);
  tmp2443 = 1.5;
  if(tmp2442 < 0.0 && tmp2443 != 0.0)
  {
    tmp2445 = modf(tmp2443, &tmp2446);
    
    if(tmp2445 > 0.5)
    {
      tmp2445 -= 1.0;
      tmp2446 += 1.0;
    }
    else if(tmp2445 < -0.5)
    {
      tmp2445 += 1.0;
      tmp2446 -= 1.0;
    }
    
    if(fabs(tmp2445) < 1e-10)
      tmp2444 = pow(tmp2442, tmp2446);
    else
    {
      tmp2448 = modf(1.0/tmp2443, &tmp2447);
      if(tmp2448 > 0.5)
      {
        tmp2448 -= 1.0;
        tmp2447 += 1.0;
      }
      else if(tmp2448 < -0.5)
      {
        tmp2448 += 1.0;
        tmp2447 -= 1.0;
      }
      if(fabs(tmp2448) < 1e-10 && ((unsigned long)tmp2447 & 1))
      {
        tmp2444 = -pow(-tmp2442, tmp2445)*pow(tmp2442, tmp2446);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2442, tmp2443);
      }
    }
  }
  else
  {
    tmp2444 = pow(tmp2442, tmp2443);
  }
  if(isnan(tmp2444) || isinf(tmp2444))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2442, tmp2443);
  }tmp2449 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2444,"(r_init[256] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2449 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[256] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2449);
    }
  }
  (data->simulationInfo->realParameter[760] /* omega_c[256] PARAM */) = sqrt(tmp2449);
  TRACE_POP
}

/*
equation index: 13491
type: SIMPLE_ASSIGN
r_init[255] = r_min + 255.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13491};
  (data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (255.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13492
type: SIMPLE_ASSIGN
omega_c[255] = sqrt(G * Md / (r_init[255] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13492};
  modelica_real tmp2450;
  modelica_real tmp2451;
  modelica_real tmp2452;
  modelica_real tmp2453;
  modelica_real tmp2454;
  modelica_real tmp2455;
  modelica_real tmp2456;
  modelica_real tmp2457;
  modelica_real tmp2458;
  modelica_real tmp2459;
  tmp2450 = (data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */);
  tmp2451 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2452 = (tmp2450 * tmp2450) + (tmp2451 * tmp2451);
  tmp2453 = 1.5;
  if(tmp2452 < 0.0 && tmp2453 != 0.0)
  {
    tmp2455 = modf(tmp2453, &tmp2456);
    
    if(tmp2455 > 0.5)
    {
      tmp2455 -= 1.0;
      tmp2456 += 1.0;
    }
    else if(tmp2455 < -0.5)
    {
      tmp2455 += 1.0;
      tmp2456 -= 1.0;
    }
    
    if(fabs(tmp2455) < 1e-10)
      tmp2454 = pow(tmp2452, tmp2456);
    else
    {
      tmp2458 = modf(1.0/tmp2453, &tmp2457);
      if(tmp2458 > 0.5)
      {
        tmp2458 -= 1.0;
        tmp2457 += 1.0;
      }
      else if(tmp2458 < -0.5)
      {
        tmp2458 += 1.0;
        tmp2457 -= 1.0;
      }
      if(fabs(tmp2458) < 1e-10 && ((unsigned long)tmp2457 & 1))
      {
        tmp2454 = -pow(-tmp2452, tmp2455)*pow(tmp2452, tmp2456);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2452, tmp2453);
      }
    }
  }
  else
  {
    tmp2454 = pow(tmp2452, tmp2453);
  }
  if(isnan(tmp2454) || isinf(tmp2454))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2452, tmp2453);
  }tmp2459 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2454,"(r_init[255] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2459 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[255] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2459);
    }
  }
  (data->simulationInfo->realParameter[759] /* omega_c[255] PARAM */) = sqrt(tmp2459);
  TRACE_POP
}

/*
equation index: 13493
type: SIMPLE_ASSIGN
r_init[254] = r_min + 254.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13493};
  (data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (254.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13494
type: SIMPLE_ASSIGN
omega_c[254] = sqrt(G * Md / (r_init[254] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13494(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13494};
  modelica_real tmp2460;
  modelica_real tmp2461;
  modelica_real tmp2462;
  modelica_real tmp2463;
  modelica_real tmp2464;
  modelica_real tmp2465;
  modelica_real tmp2466;
  modelica_real tmp2467;
  modelica_real tmp2468;
  modelica_real tmp2469;
  tmp2460 = (data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */);
  tmp2461 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2462 = (tmp2460 * tmp2460) + (tmp2461 * tmp2461);
  tmp2463 = 1.5;
  if(tmp2462 < 0.0 && tmp2463 != 0.0)
  {
    tmp2465 = modf(tmp2463, &tmp2466);
    
    if(tmp2465 > 0.5)
    {
      tmp2465 -= 1.0;
      tmp2466 += 1.0;
    }
    else if(tmp2465 < -0.5)
    {
      tmp2465 += 1.0;
      tmp2466 -= 1.0;
    }
    
    if(fabs(tmp2465) < 1e-10)
      tmp2464 = pow(tmp2462, tmp2466);
    else
    {
      tmp2468 = modf(1.0/tmp2463, &tmp2467);
      if(tmp2468 > 0.5)
      {
        tmp2468 -= 1.0;
        tmp2467 += 1.0;
      }
      else if(tmp2468 < -0.5)
      {
        tmp2468 += 1.0;
        tmp2467 -= 1.0;
      }
      if(fabs(tmp2468) < 1e-10 && ((unsigned long)tmp2467 & 1))
      {
        tmp2464 = -pow(-tmp2462, tmp2465)*pow(tmp2462, tmp2466);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2462, tmp2463);
      }
    }
  }
  else
  {
    tmp2464 = pow(tmp2462, tmp2463);
  }
  if(isnan(tmp2464) || isinf(tmp2464))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2462, tmp2463);
  }tmp2469 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2464,"(r_init[254] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2469 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[254] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2469);
    }
  }
  (data->simulationInfo->realParameter[758] /* omega_c[254] PARAM */) = sqrt(tmp2469);
  TRACE_POP
}

/*
equation index: 13495
type: SIMPLE_ASSIGN
r_init[253] = r_min + 253.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13495};
  (data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (253.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13496
type: SIMPLE_ASSIGN
omega_c[253] = sqrt(G * Md / (r_init[253] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13496(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13496};
  modelica_real tmp2470;
  modelica_real tmp2471;
  modelica_real tmp2472;
  modelica_real tmp2473;
  modelica_real tmp2474;
  modelica_real tmp2475;
  modelica_real tmp2476;
  modelica_real tmp2477;
  modelica_real tmp2478;
  modelica_real tmp2479;
  tmp2470 = (data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */);
  tmp2471 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2472 = (tmp2470 * tmp2470) + (tmp2471 * tmp2471);
  tmp2473 = 1.5;
  if(tmp2472 < 0.0 && tmp2473 != 0.0)
  {
    tmp2475 = modf(tmp2473, &tmp2476);
    
    if(tmp2475 > 0.5)
    {
      tmp2475 -= 1.0;
      tmp2476 += 1.0;
    }
    else if(tmp2475 < -0.5)
    {
      tmp2475 += 1.0;
      tmp2476 -= 1.0;
    }
    
    if(fabs(tmp2475) < 1e-10)
      tmp2474 = pow(tmp2472, tmp2476);
    else
    {
      tmp2478 = modf(1.0/tmp2473, &tmp2477);
      if(tmp2478 > 0.5)
      {
        tmp2478 -= 1.0;
        tmp2477 += 1.0;
      }
      else if(tmp2478 < -0.5)
      {
        tmp2478 += 1.0;
        tmp2477 -= 1.0;
      }
      if(fabs(tmp2478) < 1e-10 && ((unsigned long)tmp2477 & 1))
      {
        tmp2474 = -pow(-tmp2472, tmp2475)*pow(tmp2472, tmp2476);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2472, tmp2473);
      }
    }
  }
  else
  {
    tmp2474 = pow(tmp2472, tmp2473);
  }
  if(isnan(tmp2474) || isinf(tmp2474))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2472, tmp2473);
  }tmp2479 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2474,"(r_init[253] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2479 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[253] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2479);
    }
  }
  (data->simulationInfo->realParameter[757] /* omega_c[253] PARAM */) = sqrt(tmp2479);
  TRACE_POP
}

/*
equation index: 13497
type: SIMPLE_ASSIGN
r_init[252] = r_min + 252.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13497};
  (data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (252.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13498
type: SIMPLE_ASSIGN
omega_c[252] = sqrt(G * Md / (r_init[252] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13498(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13498};
  modelica_real tmp2480;
  modelica_real tmp2481;
  modelica_real tmp2482;
  modelica_real tmp2483;
  modelica_real tmp2484;
  modelica_real tmp2485;
  modelica_real tmp2486;
  modelica_real tmp2487;
  modelica_real tmp2488;
  modelica_real tmp2489;
  tmp2480 = (data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */);
  tmp2481 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2482 = (tmp2480 * tmp2480) + (tmp2481 * tmp2481);
  tmp2483 = 1.5;
  if(tmp2482 < 0.0 && tmp2483 != 0.0)
  {
    tmp2485 = modf(tmp2483, &tmp2486);
    
    if(tmp2485 > 0.5)
    {
      tmp2485 -= 1.0;
      tmp2486 += 1.0;
    }
    else if(tmp2485 < -0.5)
    {
      tmp2485 += 1.0;
      tmp2486 -= 1.0;
    }
    
    if(fabs(tmp2485) < 1e-10)
      tmp2484 = pow(tmp2482, tmp2486);
    else
    {
      tmp2488 = modf(1.0/tmp2483, &tmp2487);
      if(tmp2488 > 0.5)
      {
        tmp2488 -= 1.0;
        tmp2487 += 1.0;
      }
      else if(tmp2488 < -0.5)
      {
        tmp2488 += 1.0;
        tmp2487 -= 1.0;
      }
      if(fabs(tmp2488) < 1e-10 && ((unsigned long)tmp2487 & 1))
      {
        tmp2484 = -pow(-tmp2482, tmp2485)*pow(tmp2482, tmp2486);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2482, tmp2483);
      }
    }
  }
  else
  {
    tmp2484 = pow(tmp2482, tmp2483);
  }
  if(isnan(tmp2484) || isinf(tmp2484))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2482, tmp2483);
  }tmp2489 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2484,"(r_init[252] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2489 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[252] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2489);
    }
  }
  (data->simulationInfo->realParameter[756] /* omega_c[252] PARAM */) = sqrt(tmp2489);
  TRACE_POP
}

/*
equation index: 13499
type: SIMPLE_ASSIGN
r_init[251] = r_min + 251.0 * dr
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13499};
  (data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */) = (data->simulationInfo->realParameter[1506] /* r_min PARAM */) + (251.0) * ((data->simulationInfo->realParameter[504] /* dr PARAM */));
  TRACE_POP
}

/*
equation index: 13500
type: SIMPLE_ASSIGN
omega_c[251] = sqrt(G * Md / (r_init[251] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_13500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13500};
  modelica_real tmp2490;
  modelica_real tmp2491;
  modelica_real tmp2492;
  modelica_real tmp2493;
  modelica_real tmp2494;
  modelica_real tmp2495;
  modelica_real tmp2496;
  modelica_real tmp2497;
  modelica_real tmp2498;
  modelica_real tmp2499;
  tmp2490 = (data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */);
  tmp2491 = (data->simulationInfo->realParameter[2] /* a PARAM */) + (data->simulationInfo->realParameter[503] /* b PARAM */);
  tmp2492 = (tmp2490 * tmp2490) + (tmp2491 * tmp2491);
  tmp2493 = 1.5;
  if(tmp2492 < 0.0 && tmp2493 != 0.0)
  {
    tmp2495 = modf(tmp2493, &tmp2496);
    
    if(tmp2495 > 0.5)
    {
      tmp2495 -= 1.0;
      tmp2496 += 1.0;
    }
    else if(tmp2495 < -0.5)
    {
      tmp2495 += 1.0;
      tmp2496 -= 1.0;
    }
    
    if(fabs(tmp2495) < 1e-10)
      tmp2494 = pow(tmp2492, tmp2496);
    else
    {
      tmp2498 = modf(1.0/tmp2493, &tmp2497);
      if(tmp2498 > 0.5)
      {
        tmp2498 -= 1.0;
        tmp2497 += 1.0;
      }
      else if(tmp2498 < -0.5)
      {
        tmp2498 += 1.0;
        tmp2497 -= 1.0;
      }
      if(fabs(tmp2498) < 1e-10 && ((unsigned long)tmp2497 & 1))
      {
        tmp2494 = -pow(-tmp2492, tmp2495)*pow(tmp2492, tmp2496);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2492, tmp2493);
      }
    }
  }
  else
  {
    tmp2494 = pow(tmp2492, tmp2493);
  }
  if(isnan(tmp2494) || isinf(tmp2494))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp2492, tmp2493);
  }tmp2499 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),tmp2494,"(r_init[251] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5",equationIndexes);
  if(!(tmp2499 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / (r_init[251] ^ 2.0 + (a + b) ^ 2.0) ^ 1.5) was %g should be >= 0", tmp2499);
    }
  }
  (data->simulationInfo->realParameter[755] /* omega_c[251] PARAM */) = sqrt(tmp2499);
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_updateBoundParameters_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_13001(data, threadData);
  SpiralGalaxy_eqFunction_13002(data, threadData);
  SpiralGalaxy_eqFunction_13003(data, threadData);
  SpiralGalaxy_eqFunction_13004(data, threadData);
  SpiralGalaxy_eqFunction_13005(data, threadData);
  SpiralGalaxy_eqFunction_13006(data, threadData);
  SpiralGalaxy_eqFunction_13007(data, threadData);
  SpiralGalaxy_eqFunction_13008(data, threadData);
  SpiralGalaxy_eqFunction_13009(data, threadData);
  SpiralGalaxy_eqFunction_13010(data, threadData);
  SpiralGalaxy_eqFunction_13011(data, threadData);
  SpiralGalaxy_eqFunction_13012(data, threadData);
  SpiralGalaxy_eqFunction_13013(data, threadData);
  SpiralGalaxy_eqFunction_13014(data, threadData);
  SpiralGalaxy_eqFunction_13015(data, threadData);
  SpiralGalaxy_eqFunction_13016(data, threadData);
  SpiralGalaxy_eqFunction_13017(data, threadData);
  SpiralGalaxy_eqFunction_13018(data, threadData);
  SpiralGalaxy_eqFunction_13019(data, threadData);
  SpiralGalaxy_eqFunction_13020(data, threadData);
  SpiralGalaxy_eqFunction_13021(data, threadData);
  SpiralGalaxy_eqFunction_13022(data, threadData);
  SpiralGalaxy_eqFunction_13023(data, threadData);
  SpiralGalaxy_eqFunction_13024(data, threadData);
  SpiralGalaxy_eqFunction_13025(data, threadData);
  SpiralGalaxy_eqFunction_13026(data, threadData);
  SpiralGalaxy_eqFunction_13027(data, threadData);
  SpiralGalaxy_eqFunction_13028(data, threadData);
  SpiralGalaxy_eqFunction_13029(data, threadData);
  SpiralGalaxy_eqFunction_13030(data, threadData);
  SpiralGalaxy_eqFunction_13031(data, threadData);
  SpiralGalaxy_eqFunction_13032(data, threadData);
  SpiralGalaxy_eqFunction_13033(data, threadData);
  SpiralGalaxy_eqFunction_13034(data, threadData);
  SpiralGalaxy_eqFunction_13035(data, threadData);
  SpiralGalaxy_eqFunction_13036(data, threadData);
  SpiralGalaxy_eqFunction_13037(data, threadData);
  SpiralGalaxy_eqFunction_13038(data, threadData);
  SpiralGalaxy_eqFunction_13039(data, threadData);
  SpiralGalaxy_eqFunction_13040(data, threadData);
  SpiralGalaxy_eqFunction_13041(data, threadData);
  SpiralGalaxy_eqFunction_13042(data, threadData);
  SpiralGalaxy_eqFunction_13043(data, threadData);
  SpiralGalaxy_eqFunction_13044(data, threadData);
  SpiralGalaxy_eqFunction_13045(data, threadData);
  SpiralGalaxy_eqFunction_13046(data, threadData);
  SpiralGalaxy_eqFunction_13047(data, threadData);
  SpiralGalaxy_eqFunction_13048(data, threadData);
  SpiralGalaxy_eqFunction_13049(data, threadData);
  SpiralGalaxy_eqFunction_13050(data, threadData);
  SpiralGalaxy_eqFunction_13051(data, threadData);
  SpiralGalaxy_eqFunction_13052(data, threadData);
  SpiralGalaxy_eqFunction_13053(data, threadData);
  SpiralGalaxy_eqFunction_13054(data, threadData);
  SpiralGalaxy_eqFunction_13055(data, threadData);
  SpiralGalaxy_eqFunction_13056(data, threadData);
  SpiralGalaxy_eqFunction_13057(data, threadData);
  SpiralGalaxy_eqFunction_13058(data, threadData);
  SpiralGalaxy_eqFunction_13059(data, threadData);
  SpiralGalaxy_eqFunction_13060(data, threadData);
  SpiralGalaxy_eqFunction_13061(data, threadData);
  SpiralGalaxy_eqFunction_13062(data, threadData);
  SpiralGalaxy_eqFunction_13063(data, threadData);
  SpiralGalaxy_eqFunction_13064(data, threadData);
  SpiralGalaxy_eqFunction_13065(data, threadData);
  SpiralGalaxy_eqFunction_13066(data, threadData);
  SpiralGalaxy_eqFunction_13067(data, threadData);
  SpiralGalaxy_eqFunction_13068(data, threadData);
  SpiralGalaxy_eqFunction_13069(data, threadData);
  SpiralGalaxy_eqFunction_13070(data, threadData);
  SpiralGalaxy_eqFunction_13071(data, threadData);
  SpiralGalaxy_eqFunction_13072(data, threadData);
  SpiralGalaxy_eqFunction_13073(data, threadData);
  SpiralGalaxy_eqFunction_13074(data, threadData);
  SpiralGalaxy_eqFunction_13075(data, threadData);
  SpiralGalaxy_eqFunction_13076(data, threadData);
  SpiralGalaxy_eqFunction_13077(data, threadData);
  SpiralGalaxy_eqFunction_13078(data, threadData);
  SpiralGalaxy_eqFunction_13079(data, threadData);
  SpiralGalaxy_eqFunction_13080(data, threadData);
  SpiralGalaxy_eqFunction_13081(data, threadData);
  SpiralGalaxy_eqFunction_13082(data, threadData);
  SpiralGalaxy_eqFunction_13083(data, threadData);
  SpiralGalaxy_eqFunction_13084(data, threadData);
  SpiralGalaxy_eqFunction_13085(data, threadData);
  SpiralGalaxy_eqFunction_13086(data, threadData);
  SpiralGalaxy_eqFunction_13087(data, threadData);
  SpiralGalaxy_eqFunction_13088(data, threadData);
  SpiralGalaxy_eqFunction_13089(data, threadData);
  SpiralGalaxy_eqFunction_13090(data, threadData);
  SpiralGalaxy_eqFunction_13091(data, threadData);
  SpiralGalaxy_eqFunction_13092(data, threadData);
  SpiralGalaxy_eqFunction_13093(data, threadData);
  SpiralGalaxy_eqFunction_13094(data, threadData);
  SpiralGalaxy_eqFunction_13095(data, threadData);
  SpiralGalaxy_eqFunction_13096(data, threadData);
  SpiralGalaxy_eqFunction_13097(data, threadData);
  SpiralGalaxy_eqFunction_13098(data, threadData);
  SpiralGalaxy_eqFunction_13099(data, threadData);
  SpiralGalaxy_eqFunction_13100(data, threadData);
  SpiralGalaxy_eqFunction_13101(data, threadData);
  SpiralGalaxy_eqFunction_13102(data, threadData);
  SpiralGalaxy_eqFunction_13103(data, threadData);
  SpiralGalaxy_eqFunction_13104(data, threadData);
  SpiralGalaxy_eqFunction_13105(data, threadData);
  SpiralGalaxy_eqFunction_13106(data, threadData);
  SpiralGalaxy_eqFunction_13107(data, threadData);
  SpiralGalaxy_eqFunction_13108(data, threadData);
  SpiralGalaxy_eqFunction_13109(data, threadData);
  SpiralGalaxy_eqFunction_13110(data, threadData);
  SpiralGalaxy_eqFunction_13111(data, threadData);
  SpiralGalaxy_eqFunction_13112(data, threadData);
  SpiralGalaxy_eqFunction_13113(data, threadData);
  SpiralGalaxy_eqFunction_13114(data, threadData);
  SpiralGalaxy_eqFunction_13115(data, threadData);
  SpiralGalaxy_eqFunction_13116(data, threadData);
  SpiralGalaxy_eqFunction_13117(data, threadData);
  SpiralGalaxy_eqFunction_13118(data, threadData);
  SpiralGalaxy_eqFunction_13119(data, threadData);
  SpiralGalaxy_eqFunction_13120(data, threadData);
  SpiralGalaxy_eqFunction_13121(data, threadData);
  SpiralGalaxy_eqFunction_13122(data, threadData);
  SpiralGalaxy_eqFunction_13123(data, threadData);
  SpiralGalaxy_eqFunction_13124(data, threadData);
  SpiralGalaxy_eqFunction_13125(data, threadData);
  SpiralGalaxy_eqFunction_13126(data, threadData);
  SpiralGalaxy_eqFunction_13127(data, threadData);
  SpiralGalaxy_eqFunction_13128(data, threadData);
  SpiralGalaxy_eqFunction_13129(data, threadData);
  SpiralGalaxy_eqFunction_13130(data, threadData);
  SpiralGalaxy_eqFunction_13131(data, threadData);
  SpiralGalaxy_eqFunction_13132(data, threadData);
  SpiralGalaxy_eqFunction_13133(data, threadData);
  SpiralGalaxy_eqFunction_13134(data, threadData);
  SpiralGalaxy_eqFunction_13135(data, threadData);
  SpiralGalaxy_eqFunction_13136(data, threadData);
  SpiralGalaxy_eqFunction_13137(data, threadData);
  SpiralGalaxy_eqFunction_13138(data, threadData);
  SpiralGalaxy_eqFunction_13139(data, threadData);
  SpiralGalaxy_eqFunction_13140(data, threadData);
  SpiralGalaxy_eqFunction_13141(data, threadData);
  SpiralGalaxy_eqFunction_13142(data, threadData);
  SpiralGalaxy_eqFunction_13143(data, threadData);
  SpiralGalaxy_eqFunction_13144(data, threadData);
  SpiralGalaxy_eqFunction_13145(data, threadData);
  SpiralGalaxy_eqFunction_13146(data, threadData);
  SpiralGalaxy_eqFunction_13147(data, threadData);
  SpiralGalaxy_eqFunction_13148(data, threadData);
  SpiralGalaxy_eqFunction_13149(data, threadData);
  SpiralGalaxy_eqFunction_13150(data, threadData);
  SpiralGalaxy_eqFunction_13151(data, threadData);
  SpiralGalaxy_eqFunction_13152(data, threadData);
  SpiralGalaxy_eqFunction_13153(data, threadData);
  SpiralGalaxy_eqFunction_13154(data, threadData);
  SpiralGalaxy_eqFunction_13155(data, threadData);
  SpiralGalaxy_eqFunction_13156(data, threadData);
  SpiralGalaxy_eqFunction_13157(data, threadData);
  SpiralGalaxy_eqFunction_13158(data, threadData);
  SpiralGalaxy_eqFunction_13159(data, threadData);
  SpiralGalaxy_eqFunction_13160(data, threadData);
  SpiralGalaxy_eqFunction_13161(data, threadData);
  SpiralGalaxy_eqFunction_13162(data, threadData);
  SpiralGalaxy_eqFunction_13163(data, threadData);
  SpiralGalaxy_eqFunction_13164(data, threadData);
  SpiralGalaxy_eqFunction_13165(data, threadData);
  SpiralGalaxy_eqFunction_13166(data, threadData);
  SpiralGalaxy_eqFunction_13167(data, threadData);
  SpiralGalaxy_eqFunction_13168(data, threadData);
  SpiralGalaxy_eqFunction_13169(data, threadData);
  SpiralGalaxy_eqFunction_13170(data, threadData);
  SpiralGalaxy_eqFunction_13171(data, threadData);
  SpiralGalaxy_eqFunction_13172(data, threadData);
  SpiralGalaxy_eqFunction_13173(data, threadData);
  SpiralGalaxy_eqFunction_13174(data, threadData);
  SpiralGalaxy_eqFunction_13175(data, threadData);
  SpiralGalaxy_eqFunction_13176(data, threadData);
  SpiralGalaxy_eqFunction_13177(data, threadData);
  SpiralGalaxy_eqFunction_13178(data, threadData);
  SpiralGalaxy_eqFunction_13179(data, threadData);
  SpiralGalaxy_eqFunction_13180(data, threadData);
  SpiralGalaxy_eqFunction_13181(data, threadData);
  SpiralGalaxy_eqFunction_13182(data, threadData);
  SpiralGalaxy_eqFunction_13183(data, threadData);
  SpiralGalaxy_eqFunction_13184(data, threadData);
  SpiralGalaxy_eqFunction_13185(data, threadData);
  SpiralGalaxy_eqFunction_13186(data, threadData);
  SpiralGalaxy_eqFunction_13187(data, threadData);
  SpiralGalaxy_eqFunction_13188(data, threadData);
  SpiralGalaxy_eqFunction_13189(data, threadData);
  SpiralGalaxy_eqFunction_13190(data, threadData);
  SpiralGalaxy_eqFunction_13191(data, threadData);
  SpiralGalaxy_eqFunction_13192(data, threadData);
  SpiralGalaxy_eqFunction_13193(data, threadData);
  SpiralGalaxy_eqFunction_13194(data, threadData);
  SpiralGalaxy_eqFunction_13195(data, threadData);
  SpiralGalaxy_eqFunction_13196(data, threadData);
  SpiralGalaxy_eqFunction_13197(data, threadData);
  SpiralGalaxy_eqFunction_13198(data, threadData);
  SpiralGalaxy_eqFunction_13199(data, threadData);
  SpiralGalaxy_eqFunction_13200(data, threadData);
  SpiralGalaxy_eqFunction_13201(data, threadData);
  SpiralGalaxy_eqFunction_13202(data, threadData);
  SpiralGalaxy_eqFunction_13203(data, threadData);
  SpiralGalaxy_eqFunction_13204(data, threadData);
  SpiralGalaxy_eqFunction_13205(data, threadData);
  SpiralGalaxy_eqFunction_13206(data, threadData);
  SpiralGalaxy_eqFunction_13207(data, threadData);
  SpiralGalaxy_eqFunction_13208(data, threadData);
  SpiralGalaxy_eqFunction_13209(data, threadData);
  SpiralGalaxy_eqFunction_13210(data, threadData);
  SpiralGalaxy_eqFunction_13211(data, threadData);
  SpiralGalaxy_eqFunction_13212(data, threadData);
  SpiralGalaxy_eqFunction_13213(data, threadData);
  SpiralGalaxy_eqFunction_13214(data, threadData);
  SpiralGalaxy_eqFunction_13215(data, threadData);
  SpiralGalaxy_eqFunction_13216(data, threadData);
  SpiralGalaxy_eqFunction_13217(data, threadData);
  SpiralGalaxy_eqFunction_13218(data, threadData);
  SpiralGalaxy_eqFunction_13219(data, threadData);
  SpiralGalaxy_eqFunction_13220(data, threadData);
  SpiralGalaxy_eqFunction_13221(data, threadData);
  SpiralGalaxy_eqFunction_13222(data, threadData);
  SpiralGalaxy_eqFunction_13223(data, threadData);
  SpiralGalaxy_eqFunction_13224(data, threadData);
  SpiralGalaxy_eqFunction_13225(data, threadData);
  SpiralGalaxy_eqFunction_13226(data, threadData);
  SpiralGalaxy_eqFunction_13227(data, threadData);
  SpiralGalaxy_eqFunction_13228(data, threadData);
  SpiralGalaxy_eqFunction_13229(data, threadData);
  SpiralGalaxy_eqFunction_13230(data, threadData);
  SpiralGalaxy_eqFunction_13231(data, threadData);
  SpiralGalaxy_eqFunction_13232(data, threadData);
  SpiralGalaxy_eqFunction_13233(data, threadData);
  SpiralGalaxy_eqFunction_13234(data, threadData);
  SpiralGalaxy_eqFunction_13235(data, threadData);
  SpiralGalaxy_eqFunction_13236(data, threadData);
  SpiralGalaxy_eqFunction_13237(data, threadData);
  SpiralGalaxy_eqFunction_13238(data, threadData);
  SpiralGalaxy_eqFunction_13239(data, threadData);
  SpiralGalaxy_eqFunction_13240(data, threadData);
  SpiralGalaxy_eqFunction_13241(data, threadData);
  SpiralGalaxy_eqFunction_13242(data, threadData);
  SpiralGalaxy_eqFunction_13243(data, threadData);
  SpiralGalaxy_eqFunction_13244(data, threadData);
  SpiralGalaxy_eqFunction_13245(data, threadData);
  SpiralGalaxy_eqFunction_13246(data, threadData);
  SpiralGalaxy_eqFunction_13247(data, threadData);
  SpiralGalaxy_eqFunction_13248(data, threadData);
  SpiralGalaxy_eqFunction_13249(data, threadData);
  SpiralGalaxy_eqFunction_13250(data, threadData);
  SpiralGalaxy_eqFunction_13251(data, threadData);
  SpiralGalaxy_eqFunction_13252(data, threadData);
  SpiralGalaxy_eqFunction_13253(data, threadData);
  SpiralGalaxy_eqFunction_13254(data, threadData);
  SpiralGalaxy_eqFunction_13255(data, threadData);
  SpiralGalaxy_eqFunction_13256(data, threadData);
  SpiralGalaxy_eqFunction_13257(data, threadData);
  SpiralGalaxy_eqFunction_13258(data, threadData);
  SpiralGalaxy_eqFunction_13259(data, threadData);
  SpiralGalaxy_eqFunction_13260(data, threadData);
  SpiralGalaxy_eqFunction_13261(data, threadData);
  SpiralGalaxy_eqFunction_13262(data, threadData);
  SpiralGalaxy_eqFunction_13263(data, threadData);
  SpiralGalaxy_eqFunction_13264(data, threadData);
  SpiralGalaxy_eqFunction_13265(data, threadData);
  SpiralGalaxy_eqFunction_13266(data, threadData);
  SpiralGalaxy_eqFunction_13267(data, threadData);
  SpiralGalaxy_eqFunction_13268(data, threadData);
  SpiralGalaxy_eqFunction_13269(data, threadData);
  SpiralGalaxy_eqFunction_13270(data, threadData);
  SpiralGalaxy_eqFunction_13271(data, threadData);
  SpiralGalaxy_eqFunction_13272(data, threadData);
  SpiralGalaxy_eqFunction_13273(data, threadData);
  SpiralGalaxy_eqFunction_13274(data, threadData);
  SpiralGalaxy_eqFunction_13275(data, threadData);
  SpiralGalaxy_eqFunction_13276(data, threadData);
  SpiralGalaxy_eqFunction_13277(data, threadData);
  SpiralGalaxy_eqFunction_13278(data, threadData);
  SpiralGalaxy_eqFunction_13279(data, threadData);
  SpiralGalaxy_eqFunction_13280(data, threadData);
  SpiralGalaxy_eqFunction_13281(data, threadData);
  SpiralGalaxy_eqFunction_13282(data, threadData);
  SpiralGalaxy_eqFunction_13283(data, threadData);
  SpiralGalaxy_eqFunction_13284(data, threadData);
  SpiralGalaxy_eqFunction_13285(data, threadData);
  SpiralGalaxy_eqFunction_13286(data, threadData);
  SpiralGalaxy_eqFunction_13287(data, threadData);
  SpiralGalaxy_eqFunction_13288(data, threadData);
  SpiralGalaxy_eqFunction_13289(data, threadData);
  SpiralGalaxy_eqFunction_13290(data, threadData);
  SpiralGalaxy_eqFunction_13291(data, threadData);
  SpiralGalaxy_eqFunction_13292(data, threadData);
  SpiralGalaxy_eqFunction_13293(data, threadData);
  SpiralGalaxy_eqFunction_13294(data, threadData);
  SpiralGalaxy_eqFunction_13295(data, threadData);
  SpiralGalaxy_eqFunction_13296(data, threadData);
  SpiralGalaxy_eqFunction_13297(data, threadData);
  SpiralGalaxy_eqFunction_13298(data, threadData);
  SpiralGalaxy_eqFunction_13299(data, threadData);
  SpiralGalaxy_eqFunction_13300(data, threadData);
  SpiralGalaxy_eqFunction_13301(data, threadData);
  SpiralGalaxy_eqFunction_13302(data, threadData);
  SpiralGalaxy_eqFunction_13303(data, threadData);
  SpiralGalaxy_eqFunction_13304(data, threadData);
  SpiralGalaxy_eqFunction_13305(data, threadData);
  SpiralGalaxy_eqFunction_13306(data, threadData);
  SpiralGalaxy_eqFunction_13307(data, threadData);
  SpiralGalaxy_eqFunction_13308(data, threadData);
  SpiralGalaxy_eqFunction_13309(data, threadData);
  SpiralGalaxy_eqFunction_13310(data, threadData);
  SpiralGalaxy_eqFunction_13311(data, threadData);
  SpiralGalaxy_eqFunction_13312(data, threadData);
  SpiralGalaxy_eqFunction_13313(data, threadData);
  SpiralGalaxy_eqFunction_13314(data, threadData);
  SpiralGalaxy_eqFunction_13315(data, threadData);
  SpiralGalaxy_eqFunction_13316(data, threadData);
  SpiralGalaxy_eqFunction_13317(data, threadData);
  SpiralGalaxy_eqFunction_13318(data, threadData);
  SpiralGalaxy_eqFunction_13319(data, threadData);
  SpiralGalaxy_eqFunction_13320(data, threadData);
  SpiralGalaxy_eqFunction_13321(data, threadData);
  SpiralGalaxy_eqFunction_13322(data, threadData);
  SpiralGalaxy_eqFunction_13323(data, threadData);
  SpiralGalaxy_eqFunction_13324(data, threadData);
  SpiralGalaxy_eqFunction_13325(data, threadData);
  SpiralGalaxy_eqFunction_13326(data, threadData);
  SpiralGalaxy_eqFunction_13327(data, threadData);
  SpiralGalaxy_eqFunction_13328(data, threadData);
  SpiralGalaxy_eqFunction_13329(data, threadData);
  SpiralGalaxy_eqFunction_13330(data, threadData);
  SpiralGalaxy_eqFunction_13331(data, threadData);
  SpiralGalaxy_eqFunction_13332(data, threadData);
  SpiralGalaxy_eqFunction_13333(data, threadData);
  SpiralGalaxy_eqFunction_13334(data, threadData);
  SpiralGalaxy_eqFunction_13335(data, threadData);
  SpiralGalaxy_eqFunction_13336(data, threadData);
  SpiralGalaxy_eqFunction_13337(data, threadData);
  SpiralGalaxy_eqFunction_13338(data, threadData);
  SpiralGalaxy_eqFunction_13339(data, threadData);
  SpiralGalaxy_eqFunction_13340(data, threadData);
  SpiralGalaxy_eqFunction_13341(data, threadData);
  SpiralGalaxy_eqFunction_13342(data, threadData);
  SpiralGalaxy_eqFunction_13343(data, threadData);
  SpiralGalaxy_eqFunction_13344(data, threadData);
  SpiralGalaxy_eqFunction_13345(data, threadData);
  SpiralGalaxy_eqFunction_13346(data, threadData);
  SpiralGalaxy_eqFunction_13347(data, threadData);
  SpiralGalaxy_eqFunction_13348(data, threadData);
  SpiralGalaxy_eqFunction_13349(data, threadData);
  SpiralGalaxy_eqFunction_13350(data, threadData);
  SpiralGalaxy_eqFunction_13351(data, threadData);
  SpiralGalaxy_eqFunction_13352(data, threadData);
  SpiralGalaxy_eqFunction_13353(data, threadData);
  SpiralGalaxy_eqFunction_13354(data, threadData);
  SpiralGalaxy_eqFunction_13355(data, threadData);
  SpiralGalaxy_eqFunction_13356(data, threadData);
  SpiralGalaxy_eqFunction_13357(data, threadData);
  SpiralGalaxy_eqFunction_13358(data, threadData);
  SpiralGalaxy_eqFunction_13359(data, threadData);
  SpiralGalaxy_eqFunction_13360(data, threadData);
  SpiralGalaxy_eqFunction_13361(data, threadData);
  SpiralGalaxy_eqFunction_13362(data, threadData);
  SpiralGalaxy_eqFunction_13363(data, threadData);
  SpiralGalaxy_eqFunction_13364(data, threadData);
  SpiralGalaxy_eqFunction_13365(data, threadData);
  SpiralGalaxy_eqFunction_13366(data, threadData);
  SpiralGalaxy_eqFunction_13367(data, threadData);
  SpiralGalaxy_eqFunction_13368(data, threadData);
  SpiralGalaxy_eqFunction_13369(data, threadData);
  SpiralGalaxy_eqFunction_13370(data, threadData);
  SpiralGalaxy_eqFunction_13371(data, threadData);
  SpiralGalaxy_eqFunction_13372(data, threadData);
  SpiralGalaxy_eqFunction_13373(data, threadData);
  SpiralGalaxy_eqFunction_13374(data, threadData);
  SpiralGalaxy_eqFunction_13375(data, threadData);
  SpiralGalaxy_eqFunction_13376(data, threadData);
  SpiralGalaxy_eqFunction_13377(data, threadData);
  SpiralGalaxy_eqFunction_13378(data, threadData);
  SpiralGalaxy_eqFunction_13379(data, threadData);
  SpiralGalaxy_eqFunction_13380(data, threadData);
  SpiralGalaxy_eqFunction_13381(data, threadData);
  SpiralGalaxy_eqFunction_13382(data, threadData);
  SpiralGalaxy_eqFunction_13383(data, threadData);
  SpiralGalaxy_eqFunction_13384(data, threadData);
  SpiralGalaxy_eqFunction_13385(data, threadData);
  SpiralGalaxy_eqFunction_13386(data, threadData);
  SpiralGalaxy_eqFunction_13387(data, threadData);
  SpiralGalaxy_eqFunction_13388(data, threadData);
  SpiralGalaxy_eqFunction_13389(data, threadData);
  SpiralGalaxy_eqFunction_13390(data, threadData);
  SpiralGalaxy_eqFunction_13391(data, threadData);
  SpiralGalaxy_eqFunction_13392(data, threadData);
  SpiralGalaxy_eqFunction_13393(data, threadData);
  SpiralGalaxy_eqFunction_13394(data, threadData);
  SpiralGalaxy_eqFunction_13395(data, threadData);
  SpiralGalaxy_eqFunction_13396(data, threadData);
  SpiralGalaxy_eqFunction_13397(data, threadData);
  SpiralGalaxy_eqFunction_13398(data, threadData);
  SpiralGalaxy_eqFunction_13399(data, threadData);
  SpiralGalaxy_eqFunction_13400(data, threadData);
  SpiralGalaxy_eqFunction_13401(data, threadData);
  SpiralGalaxy_eqFunction_13402(data, threadData);
  SpiralGalaxy_eqFunction_13403(data, threadData);
  SpiralGalaxy_eqFunction_13404(data, threadData);
  SpiralGalaxy_eqFunction_13405(data, threadData);
  SpiralGalaxy_eqFunction_13406(data, threadData);
  SpiralGalaxy_eqFunction_13407(data, threadData);
  SpiralGalaxy_eqFunction_13408(data, threadData);
  SpiralGalaxy_eqFunction_13409(data, threadData);
  SpiralGalaxy_eqFunction_13410(data, threadData);
  SpiralGalaxy_eqFunction_13411(data, threadData);
  SpiralGalaxy_eqFunction_13412(data, threadData);
  SpiralGalaxy_eqFunction_13413(data, threadData);
  SpiralGalaxy_eqFunction_13414(data, threadData);
  SpiralGalaxy_eqFunction_13415(data, threadData);
  SpiralGalaxy_eqFunction_13416(data, threadData);
  SpiralGalaxy_eqFunction_13417(data, threadData);
  SpiralGalaxy_eqFunction_13418(data, threadData);
  SpiralGalaxy_eqFunction_13419(data, threadData);
  SpiralGalaxy_eqFunction_13420(data, threadData);
  SpiralGalaxy_eqFunction_13421(data, threadData);
  SpiralGalaxy_eqFunction_13422(data, threadData);
  SpiralGalaxy_eqFunction_13423(data, threadData);
  SpiralGalaxy_eqFunction_13424(data, threadData);
  SpiralGalaxy_eqFunction_13425(data, threadData);
  SpiralGalaxy_eqFunction_13426(data, threadData);
  SpiralGalaxy_eqFunction_13427(data, threadData);
  SpiralGalaxy_eqFunction_13428(data, threadData);
  SpiralGalaxy_eqFunction_13429(data, threadData);
  SpiralGalaxy_eqFunction_13430(data, threadData);
  SpiralGalaxy_eqFunction_13431(data, threadData);
  SpiralGalaxy_eqFunction_13432(data, threadData);
  SpiralGalaxy_eqFunction_13433(data, threadData);
  SpiralGalaxy_eqFunction_13434(data, threadData);
  SpiralGalaxy_eqFunction_13435(data, threadData);
  SpiralGalaxy_eqFunction_13436(data, threadData);
  SpiralGalaxy_eqFunction_13437(data, threadData);
  SpiralGalaxy_eqFunction_13438(data, threadData);
  SpiralGalaxy_eqFunction_13439(data, threadData);
  SpiralGalaxy_eqFunction_13440(data, threadData);
  SpiralGalaxy_eqFunction_13441(data, threadData);
  SpiralGalaxy_eqFunction_13442(data, threadData);
  SpiralGalaxy_eqFunction_13443(data, threadData);
  SpiralGalaxy_eqFunction_13444(data, threadData);
  SpiralGalaxy_eqFunction_13445(data, threadData);
  SpiralGalaxy_eqFunction_13446(data, threadData);
  SpiralGalaxy_eqFunction_13447(data, threadData);
  SpiralGalaxy_eqFunction_13448(data, threadData);
  SpiralGalaxy_eqFunction_13449(data, threadData);
  SpiralGalaxy_eqFunction_13450(data, threadData);
  SpiralGalaxy_eqFunction_13451(data, threadData);
  SpiralGalaxy_eqFunction_13452(data, threadData);
  SpiralGalaxy_eqFunction_13453(data, threadData);
  SpiralGalaxy_eqFunction_13454(data, threadData);
  SpiralGalaxy_eqFunction_13455(data, threadData);
  SpiralGalaxy_eqFunction_13456(data, threadData);
  SpiralGalaxy_eqFunction_13457(data, threadData);
  SpiralGalaxy_eqFunction_13458(data, threadData);
  SpiralGalaxy_eqFunction_13459(data, threadData);
  SpiralGalaxy_eqFunction_13460(data, threadData);
  SpiralGalaxy_eqFunction_13461(data, threadData);
  SpiralGalaxy_eqFunction_13462(data, threadData);
  SpiralGalaxy_eqFunction_13463(data, threadData);
  SpiralGalaxy_eqFunction_13464(data, threadData);
  SpiralGalaxy_eqFunction_13465(data, threadData);
  SpiralGalaxy_eqFunction_13466(data, threadData);
  SpiralGalaxy_eqFunction_13467(data, threadData);
  SpiralGalaxy_eqFunction_13468(data, threadData);
  SpiralGalaxy_eqFunction_13469(data, threadData);
  SpiralGalaxy_eqFunction_13470(data, threadData);
  SpiralGalaxy_eqFunction_13471(data, threadData);
  SpiralGalaxy_eqFunction_13472(data, threadData);
  SpiralGalaxy_eqFunction_13473(data, threadData);
  SpiralGalaxy_eqFunction_13474(data, threadData);
  SpiralGalaxy_eqFunction_13475(data, threadData);
  SpiralGalaxy_eqFunction_13476(data, threadData);
  SpiralGalaxy_eqFunction_13477(data, threadData);
  SpiralGalaxy_eqFunction_13478(data, threadData);
  SpiralGalaxy_eqFunction_13479(data, threadData);
  SpiralGalaxy_eqFunction_13480(data, threadData);
  SpiralGalaxy_eqFunction_13481(data, threadData);
  SpiralGalaxy_eqFunction_13482(data, threadData);
  SpiralGalaxy_eqFunction_13483(data, threadData);
  SpiralGalaxy_eqFunction_13484(data, threadData);
  SpiralGalaxy_eqFunction_13485(data, threadData);
  SpiralGalaxy_eqFunction_13486(data, threadData);
  SpiralGalaxy_eqFunction_13487(data, threadData);
  SpiralGalaxy_eqFunction_13488(data, threadData);
  SpiralGalaxy_eqFunction_13489(data, threadData);
  SpiralGalaxy_eqFunction_13490(data, threadData);
  SpiralGalaxy_eqFunction_13491(data, threadData);
  SpiralGalaxy_eqFunction_13492(data, threadData);
  SpiralGalaxy_eqFunction_13493(data, threadData);
  SpiralGalaxy_eqFunction_13494(data, threadData);
  SpiralGalaxy_eqFunction_13495(data, threadData);
  SpiralGalaxy_eqFunction_13496(data, threadData);
  SpiralGalaxy_eqFunction_13497(data, threadData);
  SpiralGalaxy_eqFunction_13498(data, threadData);
  SpiralGalaxy_eqFunction_13499(data, threadData);
  SpiralGalaxy_eqFunction_13500(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif