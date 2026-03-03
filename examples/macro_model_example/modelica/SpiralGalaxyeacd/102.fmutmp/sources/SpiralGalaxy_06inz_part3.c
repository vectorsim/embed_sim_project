#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 1501
type: SIMPLE_ASSIGN
vy[94] = cos(theta[94]) * r_init[94] * omega_c[94]
*/
void SpiralGalaxy_eqFunction_1501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[593]] /* vy[94] STATE(1) */) = (cos((data->simulationInfo->realParameter[1600] /* theta[94] PARAM */))) * (((data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */)) * ((data->simulationInfo->realParameter[598] /* omega_c[94] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8932(DATA *data, threadData_t *threadData);


/*
equation index: 1503
type: SIMPLE_ASSIGN
vz[94] = 0.0
*/
void SpiralGalaxy_eqFunction_1503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1503};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1093]] /* vz[94] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8931(DATA *data, threadData_t *threadData);


/*
equation index: 1505
type: SIMPLE_ASSIGN
z[95] = -0.024800000000000003
*/
void SpiralGalaxy_eqFunction_1505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1505};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2594]] /* z[95] STATE(1,vz[95]) */) = -0.024800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8944(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8945(DATA *data, threadData_t *threadData);


/*
equation index: 1508
type: SIMPLE_ASSIGN
y[95] = r_init[95] * sin(theta[95] - 0.0062)
*/
void SpiralGalaxy_eqFunction_1508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1508};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2094]] /* y[95] STATE(1,vy[95]) */) = ((data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */)) * (sin((data->simulationInfo->realParameter[1601] /* theta[95] PARAM */) - 0.0062));
  TRACE_POP
}

/*
equation index: 1509
type: SIMPLE_ASSIGN
x[95] = r_init[95] * cos(theta[95] - 0.0062)
*/
void SpiralGalaxy_eqFunction_1509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1594]] /* x[95] STATE(1,vx[95]) */) = ((data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */)) * (cos((data->simulationInfo->realParameter[1601] /* theta[95] PARAM */) - 0.0062));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8946(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8947(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8950(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8949(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8948(DATA *data, threadData_t *threadData);


/*
equation index: 1515
type: SIMPLE_ASSIGN
vx[95] = (-sin(theta[95])) * r_init[95] * omega_c[95]
*/
void SpiralGalaxy_eqFunction_1515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1515};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[94]] /* vx[95] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1601] /* theta[95] PARAM */)))) * (((data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */)) * ((data->simulationInfo->realParameter[599] /* omega_c[95] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8943(DATA *data, threadData_t *threadData);


/*
equation index: 1517
type: SIMPLE_ASSIGN
vy[95] = cos(theta[95]) * r_init[95] * omega_c[95]
*/
void SpiralGalaxy_eqFunction_1517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[594]] /* vy[95] STATE(1) */) = (cos((data->simulationInfo->realParameter[1601] /* theta[95] PARAM */))) * (((data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */)) * ((data->simulationInfo->realParameter[599] /* omega_c[95] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8942(DATA *data, threadData_t *threadData);


/*
equation index: 1519
type: SIMPLE_ASSIGN
vz[95] = 0.0
*/
void SpiralGalaxy_eqFunction_1519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1519};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1094]] /* vz[95] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8941(DATA *data, threadData_t *threadData);


/*
equation index: 1521
type: SIMPLE_ASSIGN
z[96] = -0.024640000000000002
*/
void SpiralGalaxy_eqFunction_1521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1521};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2595]] /* z[96] STATE(1,vz[96]) */) = -0.024640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8954(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8955(DATA *data, threadData_t *threadData);


/*
equation index: 1524
type: SIMPLE_ASSIGN
y[96] = r_init[96] * sin(theta[96] - 0.00616)
*/
void SpiralGalaxy_eqFunction_1524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1524};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2095]] /* y[96] STATE(1,vy[96]) */) = ((data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */)) * (sin((data->simulationInfo->realParameter[1602] /* theta[96] PARAM */) - 0.00616));
  TRACE_POP
}

/*
equation index: 1525
type: SIMPLE_ASSIGN
x[96] = r_init[96] * cos(theta[96] - 0.00616)
*/
void SpiralGalaxy_eqFunction_1525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1595]] /* x[96] STATE(1,vx[96]) */) = ((data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */)) * (cos((data->simulationInfo->realParameter[1602] /* theta[96] PARAM */) - 0.00616));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8956(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8957(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8960(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8959(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8958(DATA *data, threadData_t *threadData);


/*
equation index: 1531
type: SIMPLE_ASSIGN
vx[96] = (-sin(theta[96])) * r_init[96] * omega_c[96]
*/
void SpiralGalaxy_eqFunction_1531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1531};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[95]] /* vx[96] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1602] /* theta[96] PARAM */)))) * (((data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */)) * ((data->simulationInfo->realParameter[600] /* omega_c[96] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8953(DATA *data, threadData_t *threadData);


/*
equation index: 1533
type: SIMPLE_ASSIGN
vy[96] = cos(theta[96]) * r_init[96] * omega_c[96]
*/
void SpiralGalaxy_eqFunction_1533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[595]] /* vy[96] STATE(1) */) = (cos((data->simulationInfo->realParameter[1602] /* theta[96] PARAM */))) * (((data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */)) * ((data->simulationInfo->realParameter[600] /* omega_c[96] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8952(DATA *data, threadData_t *threadData);


/*
equation index: 1535
type: SIMPLE_ASSIGN
vz[96] = 0.0
*/
void SpiralGalaxy_eqFunction_1535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1535};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1095]] /* vz[96] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8951(DATA *data, threadData_t *threadData);


/*
equation index: 1537
type: SIMPLE_ASSIGN
z[97] = -0.024480000000000002
*/
void SpiralGalaxy_eqFunction_1537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1537};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2596]] /* z[97] STATE(1,vz[97]) */) = -0.024480000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8964(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8965(DATA *data, threadData_t *threadData);


/*
equation index: 1540
type: SIMPLE_ASSIGN
y[97] = r_init[97] * sin(theta[97] - 0.0061200000000000004)
*/
void SpiralGalaxy_eqFunction_1540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1540};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2096]] /* y[97] STATE(1,vy[97]) */) = ((data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */)) * (sin((data->simulationInfo->realParameter[1603] /* theta[97] PARAM */) - 0.0061200000000000004));
  TRACE_POP
}

/*
equation index: 1541
type: SIMPLE_ASSIGN
x[97] = r_init[97] * cos(theta[97] - 0.0061200000000000004)
*/
void SpiralGalaxy_eqFunction_1541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1596]] /* x[97] STATE(1,vx[97]) */) = ((data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */)) * (cos((data->simulationInfo->realParameter[1603] /* theta[97] PARAM */) - 0.0061200000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8966(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8967(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8970(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8969(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8968(DATA *data, threadData_t *threadData);


/*
equation index: 1547
type: SIMPLE_ASSIGN
vx[97] = (-sin(theta[97])) * r_init[97] * omega_c[97]
*/
void SpiralGalaxy_eqFunction_1547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1547};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[96]] /* vx[97] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1603] /* theta[97] PARAM */)))) * (((data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */)) * ((data->simulationInfo->realParameter[601] /* omega_c[97] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8963(DATA *data, threadData_t *threadData);


/*
equation index: 1549
type: SIMPLE_ASSIGN
vy[97] = cos(theta[97]) * r_init[97] * omega_c[97]
*/
void SpiralGalaxy_eqFunction_1549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[596]] /* vy[97] STATE(1) */) = (cos((data->simulationInfo->realParameter[1603] /* theta[97] PARAM */))) * (((data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */)) * ((data->simulationInfo->realParameter[601] /* omega_c[97] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8962(DATA *data, threadData_t *threadData);


/*
equation index: 1551
type: SIMPLE_ASSIGN
vz[97] = 0.0
*/
void SpiralGalaxy_eqFunction_1551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1551};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1096]] /* vz[97] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8961(DATA *data, threadData_t *threadData);


/*
equation index: 1553
type: SIMPLE_ASSIGN
z[98] = -0.02432
*/
void SpiralGalaxy_eqFunction_1553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1553};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2597]] /* z[98] STATE(1,vz[98]) */) = -0.02432;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8974(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8975(DATA *data, threadData_t *threadData);


/*
equation index: 1556
type: SIMPLE_ASSIGN
y[98] = r_init[98] * sin(theta[98] - 0.00608)
*/
void SpiralGalaxy_eqFunction_1556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1556};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2097]] /* y[98] STATE(1,vy[98]) */) = ((data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */)) * (sin((data->simulationInfo->realParameter[1604] /* theta[98] PARAM */) - 0.00608));
  TRACE_POP
}

/*
equation index: 1557
type: SIMPLE_ASSIGN
x[98] = r_init[98] * cos(theta[98] - 0.00608)
*/
void SpiralGalaxy_eqFunction_1557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1597]] /* x[98] STATE(1,vx[98]) */) = ((data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */)) * (cos((data->simulationInfo->realParameter[1604] /* theta[98] PARAM */) - 0.00608));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8976(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8977(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8980(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8979(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8978(DATA *data, threadData_t *threadData);


/*
equation index: 1563
type: SIMPLE_ASSIGN
vx[98] = (-sin(theta[98])) * r_init[98] * omega_c[98]
*/
void SpiralGalaxy_eqFunction_1563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1563};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[97]] /* vx[98] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1604] /* theta[98] PARAM */)))) * (((data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */)) * ((data->simulationInfo->realParameter[602] /* omega_c[98] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8973(DATA *data, threadData_t *threadData);


/*
equation index: 1565
type: SIMPLE_ASSIGN
vy[98] = cos(theta[98]) * r_init[98] * omega_c[98]
*/
void SpiralGalaxy_eqFunction_1565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[597]] /* vy[98] STATE(1) */) = (cos((data->simulationInfo->realParameter[1604] /* theta[98] PARAM */))) * (((data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */)) * ((data->simulationInfo->realParameter[602] /* omega_c[98] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8972(DATA *data, threadData_t *threadData);


/*
equation index: 1567
type: SIMPLE_ASSIGN
vz[98] = 0.0
*/
void SpiralGalaxy_eqFunction_1567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1567};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1097]] /* vz[98] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8971(DATA *data, threadData_t *threadData);


/*
equation index: 1569
type: SIMPLE_ASSIGN
z[99] = -0.02416
*/
void SpiralGalaxy_eqFunction_1569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1569};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2598]] /* z[99] STATE(1,vz[99]) */) = -0.02416;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8984(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8985(DATA *data, threadData_t *threadData);


/*
equation index: 1572
type: SIMPLE_ASSIGN
y[99] = r_init[99] * sin(theta[99] - 0.00604)
*/
void SpiralGalaxy_eqFunction_1572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1572};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2098]] /* y[99] STATE(1,vy[99]) */) = ((data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */)) * (sin((data->simulationInfo->realParameter[1605] /* theta[99] PARAM */) - 0.00604));
  TRACE_POP
}

/*
equation index: 1573
type: SIMPLE_ASSIGN
x[99] = r_init[99] * cos(theta[99] - 0.00604)
*/
void SpiralGalaxy_eqFunction_1573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1598]] /* x[99] STATE(1,vx[99]) */) = ((data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */)) * (cos((data->simulationInfo->realParameter[1605] /* theta[99] PARAM */) - 0.00604));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8986(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8987(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8990(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8989(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8988(DATA *data, threadData_t *threadData);


/*
equation index: 1579
type: SIMPLE_ASSIGN
vx[99] = (-sin(theta[99])) * r_init[99] * omega_c[99]
*/
void SpiralGalaxy_eqFunction_1579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1579};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[98]] /* vx[99] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1605] /* theta[99] PARAM */)))) * (((data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */)) * ((data->simulationInfo->realParameter[603] /* omega_c[99] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8983(DATA *data, threadData_t *threadData);


/*
equation index: 1581
type: SIMPLE_ASSIGN
vy[99] = cos(theta[99]) * r_init[99] * omega_c[99]
*/
void SpiralGalaxy_eqFunction_1581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[598]] /* vy[99] STATE(1) */) = (cos((data->simulationInfo->realParameter[1605] /* theta[99] PARAM */))) * (((data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */)) * ((data->simulationInfo->realParameter[603] /* omega_c[99] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8982(DATA *data, threadData_t *threadData);


/*
equation index: 1583
type: SIMPLE_ASSIGN
vz[99] = 0.0
*/
void SpiralGalaxy_eqFunction_1583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1583};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1098]] /* vz[99] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8981(DATA *data, threadData_t *threadData);


/*
equation index: 1585
type: SIMPLE_ASSIGN
z[100] = -0.024000000000000004
*/
void SpiralGalaxy_eqFunction_1585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1585};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2599]] /* z[100] STATE(1,vz[100]) */) = -0.024000000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8994(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8995(DATA *data, threadData_t *threadData);


/*
equation index: 1588
type: SIMPLE_ASSIGN
y[100] = r_init[100] * sin(theta[100] - 0.006)
*/
void SpiralGalaxy_eqFunction_1588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1588};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2099]] /* y[100] STATE(1,vy[100]) */) = ((data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */)) * (sin((data->simulationInfo->realParameter[1606] /* theta[100] PARAM */) - 0.006));
  TRACE_POP
}

/*
equation index: 1589
type: SIMPLE_ASSIGN
x[100] = r_init[100] * cos(theta[100] - 0.006)
*/
void SpiralGalaxy_eqFunction_1589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1599]] /* x[100] STATE(1,vx[100]) */) = ((data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */)) * (cos((data->simulationInfo->realParameter[1606] /* theta[100] PARAM */) - 0.006));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8996(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8997(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9000(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8999(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8998(DATA *data, threadData_t *threadData);


/*
equation index: 1595
type: SIMPLE_ASSIGN
vx[100] = (-sin(theta[100])) * r_init[100] * omega_c[100]
*/
void SpiralGalaxy_eqFunction_1595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1595};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[99]] /* vx[100] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1606] /* theta[100] PARAM */)))) * (((data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */)) * ((data->simulationInfo->realParameter[604] /* omega_c[100] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8993(DATA *data, threadData_t *threadData);


/*
equation index: 1597
type: SIMPLE_ASSIGN
vy[100] = cos(theta[100]) * r_init[100] * omega_c[100]
*/
void SpiralGalaxy_eqFunction_1597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[599]] /* vy[100] STATE(1) */) = (cos((data->simulationInfo->realParameter[1606] /* theta[100] PARAM */))) * (((data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */)) * ((data->simulationInfo->realParameter[604] /* omega_c[100] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8992(DATA *data, threadData_t *threadData);


/*
equation index: 1599
type: SIMPLE_ASSIGN
vz[100] = 0.0
*/
void SpiralGalaxy_eqFunction_1599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1599};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1099]] /* vz[100] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8991(DATA *data, threadData_t *threadData);


/*
equation index: 1601
type: SIMPLE_ASSIGN
z[101] = -0.023840000000000004
*/
void SpiralGalaxy_eqFunction_1601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1601};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2600]] /* z[101] STATE(1,vz[101]) */) = -0.023840000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9004(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9005(DATA *data, threadData_t *threadData);


/*
equation index: 1604
type: SIMPLE_ASSIGN
y[101] = r_init[101] * sin(theta[101] - 0.00596)
*/
void SpiralGalaxy_eqFunction_1604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1604};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2100]] /* y[101] STATE(1,vy[101]) */) = ((data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */)) * (sin((data->simulationInfo->realParameter[1607] /* theta[101] PARAM */) - 0.00596));
  TRACE_POP
}

/*
equation index: 1605
type: SIMPLE_ASSIGN
x[101] = r_init[101] * cos(theta[101] - 0.00596)
*/
void SpiralGalaxy_eqFunction_1605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1600]] /* x[101] STATE(1,vx[101]) */) = ((data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */)) * (cos((data->simulationInfo->realParameter[1607] /* theta[101] PARAM */) - 0.00596));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9006(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9007(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9010(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9009(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9008(DATA *data, threadData_t *threadData);


/*
equation index: 1611
type: SIMPLE_ASSIGN
vx[101] = (-sin(theta[101])) * r_init[101] * omega_c[101]
*/
void SpiralGalaxy_eqFunction_1611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1611};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[100]] /* vx[101] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1607] /* theta[101] PARAM */)))) * (((data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */)) * ((data->simulationInfo->realParameter[605] /* omega_c[101] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9003(DATA *data, threadData_t *threadData);


/*
equation index: 1613
type: SIMPLE_ASSIGN
vy[101] = cos(theta[101]) * r_init[101] * omega_c[101]
*/
void SpiralGalaxy_eqFunction_1613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[600]] /* vy[101] STATE(1) */) = (cos((data->simulationInfo->realParameter[1607] /* theta[101] PARAM */))) * (((data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */)) * ((data->simulationInfo->realParameter[605] /* omega_c[101] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9002(DATA *data, threadData_t *threadData);


/*
equation index: 1615
type: SIMPLE_ASSIGN
vz[101] = 0.0
*/
void SpiralGalaxy_eqFunction_1615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1615};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1100]] /* vz[101] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9001(DATA *data, threadData_t *threadData);


/*
equation index: 1617
type: SIMPLE_ASSIGN
z[102] = -0.023680000000000003
*/
void SpiralGalaxy_eqFunction_1617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1617};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2601]] /* z[102] STATE(1,vz[102]) */) = -0.023680000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9014(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9015(DATA *data, threadData_t *threadData);


/*
equation index: 1620
type: SIMPLE_ASSIGN
y[102] = r_init[102] * sin(theta[102] - 0.005920000000000001)
*/
void SpiralGalaxy_eqFunction_1620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1620};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2101]] /* y[102] STATE(1,vy[102]) */) = ((data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */)) * (sin((data->simulationInfo->realParameter[1608] /* theta[102] PARAM */) - 0.005920000000000001));
  TRACE_POP
}

/*
equation index: 1621
type: SIMPLE_ASSIGN
x[102] = r_init[102] * cos(theta[102] - 0.005920000000000001)
*/
void SpiralGalaxy_eqFunction_1621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1601]] /* x[102] STATE(1,vx[102]) */) = ((data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */)) * (cos((data->simulationInfo->realParameter[1608] /* theta[102] PARAM */) - 0.005920000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9016(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9017(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9020(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9019(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9018(DATA *data, threadData_t *threadData);


/*
equation index: 1627
type: SIMPLE_ASSIGN
vx[102] = (-sin(theta[102])) * r_init[102] * omega_c[102]
*/
void SpiralGalaxy_eqFunction_1627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1627};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[101]] /* vx[102] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1608] /* theta[102] PARAM */)))) * (((data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */)) * ((data->simulationInfo->realParameter[606] /* omega_c[102] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9013(DATA *data, threadData_t *threadData);


/*
equation index: 1629
type: SIMPLE_ASSIGN
vy[102] = cos(theta[102]) * r_init[102] * omega_c[102]
*/
void SpiralGalaxy_eqFunction_1629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[601]] /* vy[102] STATE(1) */) = (cos((data->simulationInfo->realParameter[1608] /* theta[102] PARAM */))) * (((data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */)) * ((data->simulationInfo->realParameter[606] /* omega_c[102] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9012(DATA *data, threadData_t *threadData);


/*
equation index: 1631
type: SIMPLE_ASSIGN
vz[102] = 0.0
*/
void SpiralGalaxy_eqFunction_1631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1631};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1101]] /* vz[102] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9011(DATA *data, threadData_t *threadData);


/*
equation index: 1633
type: SIMPLE_ASSIGN
z[103] = -0.023520000000000003
*/
void SpiralGalaxy_eqFunction_1633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1633};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2602]] /* z[103] STATE(1,vz[103]) */) = -0.023520000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9024(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9025(DATA *data, threadData_t *threadData);


/*
equation index: 1636
type: SIMPLE_ASSIGN
y[103] = r_init[103] * sin(theta[103] - 0.005880000000000001)
*/
void SpiralGalaxy_eqFunction_1636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1636};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2102]] /* y[103] STATE(1,vy[103]) */) = ((data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */)) * (sin((data->simulationInfo->realParameter[1609] /* theta[103] PARAM */) - 0.005880000000000001));
  TRACE_POP
}

/*
equation index: 1637
type: SIMPLE_ASSIGN
x[103] = r_init[103] * cos(theta[103] - 0.005880000000000001)
*/
void SpiralGalaxy_eqFunction_1637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1602]] /* x[103] STATE(1,vx[103]) */) = ((data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */)) * (cos((data->simulationInfo->realParameter[1609] /* theta[103] PARAM */) - 0.005880000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9026(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9027(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9030(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9029(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9028(DATA *data, threadData_t *threadData);


/*
equation index: 1643
type: SIMPLE_ASSIGN
vx[103] = (-sin(theta[103])) * r_init[103] * omega_c[103]
*/
void SpiralGalaxy_eqFunction_1643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1643};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[102]] /* vx[103] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1609] /* theta[103] PARAM */)))) * (((data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */)) * ((data->simulationInfo->realParameter[607] /* omega_c[103] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9023(DATA *data, threadData_t *threadData);


/*
equation index: 1645
type: SIMPLE_ASSIGN
vy[103] = cos(theta[103]) * r_init[103] * omega_c[103]
*/
void SpiralGalaxy_eqFunction_1645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[602]] /* vy[103] STATE(1) */) = (cos((data->simulationInfo->realParameter[1609] /* theta[103] PARAM */))) * (((data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */)) * ((data->simulationInfo->realParameter[607] /* omega_c[103] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9022(DATA *data, threadData_t *threadData);


/*
equation index: 1647
type: SIMPLE_ASSIGN
vz[103] = 0.0
*/
void SpiralGalaxy_eqFunction_1647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1647};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1102]] /* vz[103] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9021(DATA *data, threadData_t *threadData);


/*
equation index: 1649
type: SIMPLE_ASSIGN
z[104] = -0.02336
*/
void SpiralGalaxy_eqFunction_1649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1649};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2603]] /* z[104] STATE(1,vz[104]) */) = -0.02336;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9034(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9035(DATA *data, threadData_t *threadData);


/*
equation index: 1652
type: SIMPLE_ASSIGN
y[104] = r_init[104] * sin(theta[104] - 0.005840000000000001)
*/
void SpiralGalaxy_eqFunction_1652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1652};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2103]] /* y[104] STATE(1,vy[104]) */) = ((data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */)) * (sin((data->simulationInfo->realParameter[1610] /* theta[104] PARAM */) - 0.005840000000000001));
  TRACE_POP
}

/*
equation index: 1653
type: SIMPLE_ASSIGN
x[104] = r_init[104] * cos(theta[104] - 0.005840000000000001)
*/
void SpiralGalaxy_eqFunction_1653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1603]] /* x[104] STATE(1,vx[104]) */) = ((data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */)) * (cos((data->simulationInfo->realParameter[1610] /* theta[104] PARAM */) - 0.005840000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9036(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9037(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9040(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9039(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9038(DATA *data, threadData_t *threadData);


/*
equation index: 1659
type: SIMPLE_ASSIGN
vx[104] = (-sin(theta[104])) * r_init[104] * omega_c[104]
*/
void SpiralGalaxy_eqFunction_1659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1659};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[103]] /* vx[104] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1610] /* theta[104] PARAM */)))) * (((data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */)) * ((data->simulationInfo->realParameter[608] /* omega_c[104] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9033(DATA *data, threadData_t *threadData);


/*
equation index: 1661
type: SIMPLE_ASSIGN
vy[104] = cos(theta[104]) * r_init[104] * omega_c[104]
*/
void SpiralGalaxy_eqFunction_1661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[603]] /* vy[104] STATE(1) */) = (cos((data->simulationInfo->realParameter[1610] /* theta[104] PARAM */))) * (((data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */)) * ((data->simulationInfo->realParameter[608] /* omega_c[104] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9032(DATA *data, threadData_t *threadData);


/*
equation index: 1663
type: SIMPLE_ASSIGN
vz[104] = 0.0
*/
void SpiralGalaxy_eqFunction_1663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1663};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1103]] /* vz[104] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9031(DATA *data, threadData_t *threadData);


/*
equation index: 1665
type: SIMPLE_ASSIGN
z[105] = -0.023200000000000002
*/
void SpiralGalaxy_eqFunction_1665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1665};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2604]] /* z[105] STATE(1,vz[105]) */) = -0.023200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9044(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9045(DATA *data, threadData_t *threadData);


/*
equation index: 1668
type: SIMPLE_ASSIGN
y[105] = r_init[105] * sin(theta[105] - 0.0058000000000000005)
*/
void SpiralGalaxy_eqFunction_1668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1668};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2104]] /* y[105] STATE(1,vy[105]) */) = ((data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */)) * (sin((data->simulationInfo->realParameter[1611] /* theta[105] PARAM */) - 0.0058000000000000005));
  TRACE_POP
}

/*
equation index: 1669
type: SIMPLE_ASSIGN
x[105] = r_init[105] * cos(theta[105] - 0.0058000000000000005)
*/
void SpiralGalaxy_eqFunction_1669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1604]] /* x[105] STATE(1,vx[105]) */) = ((data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */)) * (cos((data->simulationInfo->realParameter[1611] /* theta[105] PARAM */) - 0.0058000000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9046(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9047(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9050(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9049(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9048(DATA *data, threadData_t *threadData);


/*
equation index: 1675
type: SIMPLE_ASSIGN
vx[105] = (-sin(theta[105])) * r_init[105] * omega_c[105]
*/
void SpiralGalaxy_eqFunction_1675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1675};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[104]] /* vx[105] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1611] /* theta[105] PARAM */)))) * (((data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */)) * ((data->simulationInfo->realParameter[609] /* omega_c[105] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9043(DATA *data, threadData_t *threadData);


/*
equation index: 1677
type: SIMPLE_ASSIGN
vy[105] = cos(theta[105]) * r_init[105] * omega_c[105]
*/
void SpiralGalaxy_eqFunction_1677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[604]] /* vy[105] STATE(1) */) = (cos((data->simulationInfo->realParameter[1611] /* theta[105] PARAM */))) * (((data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */)) * ((data->simulationInfo->realParameter[609] /* omega_c[105] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9042(DATA *data, threadData_t *threadData);


/*
equation index: 1679
type: SIMPLE_ASSIGN
vz[105] = 0.0
*/
void SpiralGalaxy_eqFunction_1679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1679};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1104]] /* vz[105] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9041(DATA *data, threadData_t *threadData);


/*
equation index: 1681
type: SIMPLE_ASSIGN
z[106] = -0.023040000000000005
*/
void SpiralGalaxy_eqFunction_1681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1681};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2605]] /* z[106] STATE(1,vz[106]) */) = -0.023040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9054(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9055(DATA *data, threadData_t *threadData);


/*
equation index: 1684
type: SIMPLE_ASSIGN
y[106] = r_init[106] * sin(theta[106] - 0.00576)
*/
void SpiralGalaxy_eqFunction_1684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1684};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2105]] /* y[106] STATE(1,vy[106]) */) = ((data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */)) * (sin((data->simulationInfo->realParameter[1612] /* theta[106] PARAM */) - 0.00576));
  TRACE_POP
}

/*
equation index: 1685
type: SIMPLE_ASSIGN
x[106] = r_init[106] * cos(theta[106] - 0.00576)
*/
void SpiralGalaxy_eqFunction_1685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1605]] /* x[106] STATE(1,vx[106]) */) = ((data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */)) * (cos((data->simulationInfo->realParameter[1612] /* theta[106] PARAM */) - 0.00576));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9056(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9057(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9060(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9059(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9058(DATA *data, threadData_t *threadData);


/*
equation index: 1691
type: SIMPLE_ASSIGN
vx[106] = (-sin(theta[106])) * r_init[106] * omega_c[106]
*/
void SpiralGalaxy_eqFunction_1691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1691};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[105]] /* vx[106] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1612] /* theta[106] PARAM */)))) * (((data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */)) * ((data->simulationInfo->realParameter[610] /* omega_c[106] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9053(DATA *data, threadData_t *threadData);


/*
equation index: 1693
type: SIMPLE_ASSIGN
vy[106] = cos(theta[106]) * r_init[106] * omega_c[106]
*/
void SpiralGalaxy_eqFunction_1693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[605]] /* vy[106] STATE(1) */) = (cos((data->simulationInfo->realParameter[1612] /* theta[106] PARAM */))) * (((data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */)) * ((data->simulationInfo->realParameter[610] /* omega_c[106] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9052(DATA *data, threadData_t *threadData);


/*
equation index: 1695
type: SIMPLE_ASSIGN
vz[106] = 0.0
*/
void SpiralGalaxy_eqFunction_1695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1695};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1105]] /* vz[106] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9051(DATA *data, threadData_t *threadData);


/*
equation index: 1697
type: SIMPLE_ASSIGN
z[107] = -0.022880000000000005
*/
void SpiralGalaxy_eqFunction_1697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1697};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2606]] /* z[107] STATE(1,vz[107]) */) = -0.022880000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9064(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9065(DATA *data, threadData_t *threadData);


/*
equation index: 1700
type: SIMPLE_ASSIGN
y[107] = r_init[107] * sin(theta[107] - 0.005720000000000001)
*/
void SpiralGalaxy_eqFunction_1700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1700};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2106]] /* y[107] STATE(1,vy[107]) */) = ((data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */)) * (sin((data->simulationInfo->realParameter[1613] /* theta[107] PARAM */) - 0.005720000000000001));
  TRACE_POP
}

/*
equation index: 1701
type: SIMPLE_ASSIGN
x[107] = r_init[107] * cos(theta[107] - 0.005720000000000001)
*/
void SpiralGalaxy_eqFunction_1701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1606]] /* x[107] STATE(1,vx[107]) */) = ((data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */)) * (cos((data->simulationInfo->realParameter[1613] /* theta[107] PARAM */) - 0.005720000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9066(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9067(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9070(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9069(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9068(DATA *data, threadData_t *threadData);


/*
equation index: 1707
type: SIMPLE_ASSIGN
vx[107] = (-sin(theta[107])) * r_init[107] * omega_c[107]
*/
void SpiralGalaxy_eqFunction_1707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1707};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[106]] /* vx[107] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1613] /* theta[107] PARAM */)))) * (((data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */)) * ((data->simulationInfo->realParameter[611] /* omega_c[107] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9063(DATA *data, threadData_t *threadData);


/*
equation index: 1709
type: SIMPLE_ASSIGN
vy[107] = cos(theta[107]) * r_init[107] * omega_c[107]
*/
void SpiralGalaxy_eqFunction_1709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[606]] /* vy[107] STATE(1) */) = (cos((data->simulationInfo->realParameter[1613] /* theta[107] PARAM */))) * (((data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */)) * ((data->simulationInfo->realParameter[611] /* omega_c[107] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9062(DATA *data, threadData_t *threadData);


/*
equation index: 1711
type: SIMPLE_ASSIGN
vz[107] = 0.0
*/
void SpiralGalaxy_eqFunction_1711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1711};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1106]] /* vz[107] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9061(DATA *data, threadData_t *threadData);


/*
equation index: 1713
type: SIMPLE_ASSIGN
z[108] = -0.022720000000000004
*/
void SpiralGalaxy_eqFunction_1713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1713};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2607]] /* z[108] STATE(1,vz[108]) */) = -0.022720000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9074(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9075(DATA *data, threadData_t *threadData);


/*
equation index: 1716
type: SIMPLE_ASSIGN
y[108] = r_init[108] * sin(theta[108] - 0.005680000000000001)
*/
void SpiralGalaxy_eqFunction_1716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1716};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2107]] /* y[108] STATE(1,vy[108]) */) = ((data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */)) * (sin((data->simulationInfo->realParameter[1614] /* theta[108] PARAM */) - 0.005680000000000001));
  TRACE_POP
}

/*
equation index: 1717
type: SIMPLE_ASSIGN
x[108] = r_init[108] * cos(theta[108] - 0.005680000000000001)
*/
void SpiralGalaxy_eqFunction_1717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1607]] /* x[108] STATE(1,vx[108]) */) = ((data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */)) * (cos((data->simulationInfo->realParameter[1614] /* theta[108] PARAM */) - 0.005680000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9076(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9077(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9080(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9079(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9078(DATA *data, threadData_t *threadData);


/*
equation index: 1723
type: SIMPLE_ASSIGN
vx[108] = (-sin(theta[108])) * r_init[108] * omega_c[108]
*/
void SpiralGalaxy_eqFunction_1723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1723};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[107]] /* vx[108] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1614] /* theta[108] PARAM */)))) * (((data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */)) * ((data->simulationInfo->realParameter[612] /* omega_c[108] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9073(DATA *data, threadData_t *threadData);


/*
equation index: 1725
type: SIMPLE_ASSIGN
vy[108] = cos(theta[108]) * r_init[108] * omega_c[108]
*/
void SpiralGalaxy_eqFunction_1725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[607]] /* vy[108] STATE(1) */) = (cos((data->simulationInfo->realParameter[1614] /* theta[108] PARAM */))) * (((data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */)) * ((data->simulationInfo->realParameter[612] /* omega_c[108] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9072(DATA *data, threadData_t *threadData);


/*
equation index: 1727
type: SIMPLE_ASSIGN
vz[108] = 0.0
*/
void SpiralGalaxy_eqFunction_1727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1727};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1107]] /* vz[108] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9071(DATA *data, threadData_t *threadData);


/*
equation index: 1729
type: SIMPLE_ASSIGN
z[109] = -0.02256
*/
void SpiralGalaxy_eqFunction_1729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1729};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2608]] /* z[109] STATE(1,vz[109]) */) = -0.02256;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9084(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9085(DATA *data, threadData_t *threadData);


/*
equation index: 1732
type: SIMPLE_ASSIGN
y[109] = r_init[109] * sin(theta[109] - 0.005640000000000001)
*/
void SpiralGalaxy_eqFunction_1732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1732};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2108]] /* y[109] STATE(1,vy[109]) */) = ((data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */)) * (sin((data->simulationInfo->realParameter[1615] /* theta[109] PARAM */) - 0.005640000000000001));
  TRACE_POP
}

/*
equation index: 1733
type: SIMPLE_ASSIGN
x[109] = r_init[109] * cos(theta[109] - 0.005640000000000001)
*/
void SpiralGalaxy_eqFunction_1733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1608]] /* x[109] STATE(1,vx[109]) */) = ((data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */)) * (cos((data->simulationInfo->realParameter[1615] /* theta[109] PARAM */) - 0.005640000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9086(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9087(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9090(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9089(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9088(DATA *data, threadData_t *threadData);


/*
equation index: 1739
type: SIMPLE_ASSIGN
vx[109] = (-sin(theta[109])) * r_init[109] * omega_c[109]
*/
void SpiralGalaxy_eqFunction_1739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1739};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[108]] /* vx[109] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1615] /* theta[109] PARAM */)))) * (((data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */)) * ((data->simulationInfo->realParameter[613] /* omega_c[109] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9083(DATA *data, threadData_t *threadData);


/*
equation index: 1741
type: SIMPLE_ASSIGN
vy[109] = cos(theta[109]) * r_init[109] * omega_c[109]
*/
void SpiralGalaxy_eqFunction_1741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[608]] /* vy[109] STATE(1) */) = (cos((data->simulationInfo->realParameter[1615] /* theta[109] PARAM */))) * (((data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */)) * ((data->simulationInfo->realParameter[613] /* omega_c[109] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9082(DATA *data, threadData_t *threadData);


/*
equation index: 1743
type: SIMPLE_ASSIGN
vz[109] = 0.0
*/
void SpiralGalaxy_eqFunction_1743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1743};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1108]] /* vz[109] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9081(DATA *data, threadData_t *threadData);


/*
equation index: 1745
type: SIMPLE_ASSIGN
z[110] = -0.022400000000000003
*/
void SpiralGalaxy_eqFunction_1745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1745};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2609]] /* z[110] STATE(1,vz[110]) */) = -0.022400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9094(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9095(DATA *data, threadData_t *threadData);


/*
equation index: 1748
type: SIMPLE_ASSIGN
y[110] = r_init[110] * sin(theta[110] - 0.005600000000000001)
*/
void SpiralGalaxy_eqFunction_1748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1748};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2109]] /* y[110] STATE(1,vy[110]) */) = ((data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */)) * (sin((data->simulationInfo->realParameter[1616] /* theta[110] PARAM */) - 0.005600000000000001));
  TRACE_POP
}

/*
equation index: 1749
type: SIMPLE_ASSIGN
x[110] = r_init[110] * cos(theta[110] - 0.005600000000000001)
*/
void SpiralGalaxy_eqFunction_1749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1609]] /* x[110] STATE(1,vx[110]) */) = ((data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */)) * (cos((data->simulationInfo->realParameter[1616] /* theta[110] PARAM */) - 0.005600000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9096(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9097(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9100(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9099(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9098(DATA *data, threadData_t *threadData);


/*
equation index: 1755
type: SIMPLE_ASSIGN
vx[110] = (-sin(theta[110])) * r_init[110] * omega_c[110]
*/
void SpiralGalaxy_eqFunction_1755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1755};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[109]] /* vx[110] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1616] /* theta[110] PARAM */)))) * (((data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */)) * ((data->simulationInfo->realParameter[614] /* omega_c[110] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9093(DATA *data, threadData_t *threadData);


/*
equation index: 1757
type: SIMPLE_ASSIGN
vy[110] = cos(theta[110]) * r_init[110] * omega_c[110]
*/
void SpiralGalaxy_eqFunction_1757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[609]] /* vy[110] STATE(1) */) = (cos((data->simulationInfo->realParameter[1616] /* theta[110] PARAM */))) * (((data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */)) * ((data->simulationInfo->realParameter[614] /* omega_c[110] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9092(DATA *data, threadData_t *threadData);


/*
equation index: 1759
type: SIMPLE_ASSIGN
vz[110] = 0.0
*/
void SpiralGalaxy_eqFunction_1759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1759};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1109]] /* vz[110] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9091(DATA *data, threadData_t *threadData);


/*
equation index: 1761
type: SIMPLE_ASSIGN
z[111] = -0.022240000000000003
*/
void SpiralGalaxy_eqFunction_1761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1761};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2610]] /* z[111] STATE(1,vz[111]) */) = -0.022240000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9104(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9105(DATA *data, threadData_t *threadData);


/*
equation index: 1764
type: SIMPLE_ASSIGN
y[111] = r_init[111] * sin(theta[111] - 0.005560000000000001)
*/
void SpiralGalaxy_eqFunction_1764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1764};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2110]] /* y[111] STATE(1,vy[111]) */) = ((data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */)) * (sin((data->simulationInfo->realParameter[1617] /* theta[111] PARAM */) - 0.005560000000000001));
  TRACE_POP
}

/*
equation index: 1765
type: SIMPLE_ASSIGN
x[111] = r_init[111] * cos(theta[111] - 0.005560000000000001)
*/
void SpiralGalaxy_eqFunction_1765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1610]] /* x[111] STATE(1,vx[111]) */) = ((data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */)) * (cos((data->simulationInfo->realParameter[1617] /* theta[111] PARAM */) - 0.005560000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9106(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9107(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9110(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9109(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9108(DATA *data, threadData_t *threadData);


/*
equation index: 1771
type: SIMPLE_ASSIGN
vx[111] = (-sin(theta[111])) * r_init[111] * omega_c[111]
*/
void SpiralGalaxy_eqFunction_1771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1771};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[110]] /* vx[111] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1617] /* theta[111] PARAM */)))) * (((data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */)) * ((data->simulationInfo->realParameter[615] /* omega_c[111] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9103(DATA *data, threadData_t *threadData);


/*
equation index: 1773
type: SIMPLE_ASSIGN
vy[111] = cos(theta[111]) * r_init[111] * omega_c[111]
*/
void SpiralGalaxy_eqFunction_1773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[610]] /* vy[111] STATE(1) */) = (cos((data->simulationInfo->realParameter[1617] /* theta[111] PARAM */))) * (((data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */)) * ((data->simulationInfo->realParameter[615] /* omega_c[111] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9102(DATA *data, threadData_t *threadData);


/*
equation index: 1775
type: SIMPLE_ASSIGN
vz[111] = 0.0
*/
void SpiralGalaxy_eqFunction_1775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1775};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1110]] /* vz[111] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9101(DATA *data, threadData_t *threadData);


/*
equation index: 1777
type: SIMPLE_ASSIGN
z[112] = -0.022080000000000002
*/
void SpiralGalaxy_eqFunction_1777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1777};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2611]] /* z[112] STATE(1,vz[112]) */) = -0.022080000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9114(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9115(DATA *data, threadData_t *threadData);


/*
equation index: 1780
type: SIMPLE_ASSIGN
y[112] = r_init[112] * sin(theta[112] - 0.005520000000000001)
*/
void SpiralGalaxy_eqFunction_1780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1780};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2111]] /* y[112] STATE(1,vy[112]) */) = ((data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */)) * (sin((data->simulationInfo->realParameter[1618] /* theta[112] PARAM */) - 0.005520000000000001));
  TRACE_POP
}

/*
equation index: 1781
type: SIMPLE_ASSIGN
x[112] = r_init[112] * cos(theta[112] - 0.005520000000000001)
*/
void SpiralGalaxy_eqFunction_1781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1611]] /* x[112] STATE(1,vx[112]) */) = ((data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */)) * (cos((data->simulationInfo->realParameter[1618] /* theta[112] PARAM */) - 0.005520000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9116(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9117(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9120(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9119(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9118(DATA *data, threadData_t *threadData);


/*
equation index: 1787
type: SIMPLE_ASSIGN
vx[112] = (-sin(theta[112])) * r_init[112] * omega_c[112]
*/
void SpiralGalaxy_eqFunction_1787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1787};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[111]] /* vx[112] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1618] /* theta[112] PARAM */)))) * (((data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */)) * ((data->simulationInfo->realParameter[616] /* omega_c[112] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9113(DATA *data, threadData_t *threadData);


/*
equation index: 1789
type: SIMPLE_ASSIGN
vy[112] = cos(theta[112]) * r_init[112] * omega_c[112]
*/
void SpiralGalaxy_eqFunction_1789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[611]] /* vy[112] STATE(1) */) = (cos((data->simulationInfo->realParameter[1618] /* theta[112] PARAM */))) * (((data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */)) * ((data->simulationInfo->realParameter[616] /* omega_c[112] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9112(DATA *data, threadData_t *threadData);


/*
equation index: 1791
type: SIMPLE_ASSIGN
vz[112] = 0.0
*/
void SpiralGalaxy_eqFunction_1791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1791};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1111]] /* vz[112] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9111(DATA *data, threadData_t *threadData);


/*
equation index: 1793
type: SIMPLE_ASSIGN
z[113] = -0.021920000000000002
*/
void SpiralGalaxy_eqFunction_1793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1793};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2612]] /* z[113] STATE(1,vz[113]) */) = -0.021920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9124(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9125(DATA *data, threadData_t *threadData);


/*
equation index: 1796
type: SIMPLE_ASSIGN
y[113] = r_init[113] * sin(theta[113] - 0.0054800000000000005)
*/
void SpiralGalaxy_eqFunction_1796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1796};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2112]] /* y[113] STATE(1,vy[113]) */) = ((data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */)) * (sin((data->simulationInfo->realParameter[1619] /* theta[113] PARAM */) - 0.0054800000000000005));
  TRACE_POP
}

/*
equation index: 1797
type: SIMPLE_ASSIGN
x[113] = r_init[113] * cos(theta[113] - 0.0054800000000000005)
*/
void SpiralGalaxy_eqFunction_1797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1612]] /* x[113] STATE(1,vx[113]) */) = ((data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */)) * (cos((data->simulationInfo->realParameter[1619] /* theta[113] PARAM */) - 0.0054800000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9126(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9127(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9130(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9129(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9128(DATA *data, threadData_t *threadData);


/*
equation index: 1803
type: SIMPLE_ASSIGN
vx[113] = (-sin(theta[113])) * r_init[113] * omega_c[113]
*/
void SpiralGalaxy_eqFunction_1803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1803};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[112]] /* vx[113] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1619] /* theta[113] PARAM */)))) * (((data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */)) * ((data->simulationInfo->realParameter[617] /* omega_c[113] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9123(DATA *data, threadData_t *threadData);


/*
equation index: 1805
type: SIMPLE_ASSIGN
vy[113] = cos(theta[113]) * r_init[113] * omega_c[113]
*/
void SpiralGalaxy_eqFunction_1805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[612]] /* vy[113] STATE(1) */) = (cos((data->simulationInfo->realParameter[1619] /* theta[113] PARAM */))) * (((data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */)) * ((data->simulationInfo->realParameter[617] /* omega_c[113] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9122(DATA *data, threadData_t *threadData);


/*
equation index: 1807
type: SIMPLE_ASSIGN
vz[113] = 0.0
*/
void SpiralGalaxy_eqFunction_1807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1807};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1112]] /* vz[113] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9121(DATA *data, threadData_t *threadData);


/*
equation index: 1809
type: SIMPLE_ASSIGN
z[114] = -0.02176
*/
void SpiralGalaxy_eqFunction_1809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1809};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2613]] /* z[114] STATE(1,vz[114]) */) = -0.02176;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9134(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9135(DATA *data, threadData_t *threadData);


/*
equation index: 1812
type: SIMPLE_ASSIGN
y[114] = r_init[114] * sin(theta[114] - 0.00544)
*/
void SpiralGalaxy_eqFunction_1812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1812};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2113]] /* y[114] STATE(1,vy[114]) */) = ((data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */)) * (sin((data->simulationInfo->realParameter[1620] /* theta[114] PARAM */) - 0.00544));
  TRACE_POP
}

/*
equation index: 1813
type: SIMPLE_ASSIGN
x[114] = r_init[114] * cos(theta[114] - 0.00544)
*/
void SpiralGalaxy_eqFunction_1813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1613]] /* x[114] STATE(1,vx[114]) */) = ((data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */)) * (cos((data->simulationInfo->realParameter[1620] /* theta[114] PARAM */) - 0.00544));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9136(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9137(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9140(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9139(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9138(DATA *data, threadData_t *threadData);


/*
equation index: 1819
type: SIMPLE_ASSIGN
vx[114] = (-sin(theta[114])) * r_init[114] * omega_c[114]
*/
void SpiralGalaxy_eqFunction_1819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1819};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[113]] /* vx[114] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1620] /* theta[114] PARAM */)))) * (((data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */)) * ((data->simulationInfo->realParameter[618] /* omega_c[114] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9133(DATA *data, threadData_t *threadData);


/*
equation index: 1821
type: SIMPLE_ASSIGN
vy[114] = cos(theta[114]) * r_init[114] * omega_c[114]
*/
void SpiralGalaxy_eqFunction_1821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[613]] /* vy[114] STATE(1) */) = (cos((data->simulationInfo->realParameter[1620] /* theta[114] PARAM */))) * (((data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */)) * ((data->simulationInfo->realParameter[618] /* omega_c[114] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9132(DATA *data, threadData_t *threadData);


/*
equation index: 1823
type: SIMPLE_ASSIGN
vz[114] = 0.0
*/
void SpiralGalaxy_eqFunction_1823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1823};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1113]] /* vz[114] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9131(DATA *data, threadData_t *threadData);


/*
equation index: 1825
type: SIMPLE_ASSIGN
z[115] = -0.021600000000000005
*/
void SpiralGalaxy_eqFunction_1825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1825};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2614]] /* z[115] STATE(1,vz[115]) */) = -0.021600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9144(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9145(DATA *data, threadData_t *threadData);


/*
equation index: 1828
type: SIMPLE_ASSIGN
y[115] = r_init[115] * sin(theta[115] - 0.0054)
*/
void SpiralGalaxy_eqFunction_1828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1828};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2114]] /* y[115] STATE(1,vy[115]) */) = ((data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */)) * (sin((data->simulationInfo->realParameter[1621] /* theta[115] PARAM */) - 0.0054));
  TRACE_POP
}

/*
equation index: 1829
type: SIMPLE_ASSIGN
x[115] = r_init[115] * cos(theta[115] - 0.0054)
*/
void SpiralGalaxy_eqFunction_1829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1614]] /* x[115] STATE(1,vx[115]) */) = ((data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */)) * (cos((data->simulationInfo->realParameter[1621] /* theta[115] PARAM */) - 0.0054));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9146(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9147(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9150(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9149(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9148(DATA *data, threadData_t *threadData);


/*
equation index: 1835
type: SIMPLE_ASSIGN
vx[115] = (-sin(theta[115])) * r_init[115] * omega_c[115]
*/
void SpiralGalaxy_eqFunction_1835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1835};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[114]] /* vx[115] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1621] /* theta[115] PARAM */)))) * (((data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */)) * ((data->simulationInfo->realParameter[619] /* omega_c[115] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9143(DATA *data, threadData_t *threadData);


/*
equation index: 1837
type: SIMPLE_ASSIGN
vy[115] = cos(theta[115]) * r_init[115] * omega_c[115]
*/
void SpiralGalaxy_eqFunction_1837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[614]] /* vy[115] STATE(1) */) = (cos((data->simulationInfo->realParameter[1621] /* theta[115] PARAM */))) * (((data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */)) * ((data->simulationInfo->realParameter[619] /* omega_c[115] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9142(DATA *data, threadData_t *threadData);


/*
equation index: 1839
type: SIMPLE_ASSIGN
vz[115] = 0.0
*/
void SpiralGalaxy_eqFunction_1839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1839};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1114]] /* vz[115] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9141(DATA *data, threadData_t *threadData);


/*
equation index: 1841
type: SIMPLE_ASSIGN
z[116] = -0.02144
*/
void SpiralGalaxy_eqFunction_1841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1841};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2615]] /* z[116] STATE(1,vz[116]) */) = -0.02144;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9154(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9155(DATA *data, threadData_t *threadData);


/*
equation index: 1844
type: SIMPLE_ASSIGN
y[116] = r_init[116] * sin(theta[116] - 0.00536)
*/
void SpiralGalaxy_eqFunction_1844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1844};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2115]] /* y[116] STATE(1,vy[116]) */) = ((data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */)) * (sin((data->simulationInfo->realParameter[1622] /* theta[116] PARAM */) - 0.00536));
  TRACE_POP
}

/*
equation index: 1845
type: SIMPLE_ASSIGN
x[116] = r_init[116] * cos(theta[116] - 0.00536)
*/
void SpiralGalaxy_eqFunction_1845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1615]] /* x[116] STATE(1,vx[116]) */) = ((data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */)) * (cos((data->simulationInfo->realParameter[1622] /* theta[116] PARAM */) - 0.00536));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9156(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9157(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9160(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9159(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9158(DATA *data, threadData_t *threadData);


/*
equation index: 1851
type: SIMPLE_ASSIGN
vx[116] = (-sin(theta[116])) * r_init[116] * omega_c[116]
*/
void SpiralGalaxy_eqFunction_1851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1851};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[115]] /* vx[116] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1622] /* theta[116] PARAM */)))) * (((data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */)) * ((data->simulationInfo->realParameter[620] /* omega_c[116] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9153(DATA *data, threadData_t *threadData);


/*
equation index: 1853
type: SIMPLE_ASSIGN
vy[116] = cos(theta[116]) * r_init[116] * omega_c[116]
*/
void SpiralGalaxy_eqFunction_1853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[615]] /* vy[116] STATE(1) */) = (cos((data->simulationInfo->realParameter[1622] /* theta[116] PARAM */))) * (((data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */)) * ((data->simulationInfo->realParameter[620] /* omega_c[116] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9152(DATA *data, threadData_t *threadData);


/*
equation index: 1855
type: SIMPLE_ASSIGN
vz[116] = 0.0
*/
void SpiralGalaxy_eqFunction_1855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1855};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1115]] /* vz[116] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9151(DATA *data, threadData_t *threadData);


/*
equation index: 1857
type: SIMPLE_ASSIGN
z[117] = -0.02128
*/
void SpiralGalaxy_eqFunction_1857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1857};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2616]] /* z[117] STATE(1,vz[117]) */) = -0.02128;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9164(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9165(DATA *data, threadData_t *threadData);


/*
equation index: 1860
type: SIMPLE_ASSIGN
y[117] = r_init[117] * sin(theta[117] - 0.00532)
*/
void SpiralGalaxy_eqFunction_1860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1860};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2116]] /* y[117] STATE(1,vy[117]) */) = ((data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */)) * (sin((data->simulationInfo->realParameter[1623] /* theta[117] PARAM */) - 0.00532));
  TRACE_POP
}

/*
equation index: 1861
type: SIMPLE_ASSIGN
x[117] = r_init[117] * cos(theta[117] - 0.00532)
*/
void SpiralGalaxy_eqFunction_1861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1616]] /* x[117] STATE(1,vx[117]) */) = ((data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */)) * (cos((data->simulationInfo->realParameter[1623] /* theta[117] PARAM */) - 0.00532));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9166(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9167(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9170(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9169(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9168(DATA *data, threadData_t *threadData);


/*
equation index: 1867
type: SIMPLE_ASSIGN
vx[117] = (-sin(theta[117])) * r_init[117] * omega_c[117]
*/
void SpiralGalaxy_eqFunction_1867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1867};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[116]] /* vx[117] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1623] /* theta[117] PARAM */)))) * (((data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */)) * ((data->simulationInfo->realParameter[621] /* omega_c[117] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9163(DATA *data, threadData_t *threadData);


/*
equation index: 1869
type: SIMPLE_ASSIGN
vy[117] = cos(theta[117]) * r_init[117] * omega_c[117]
*/
void SpiralGalaxy_eqFunction_1869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[616]] /* vy[117] STATE(1) */) = (cos((data->simulationInfo->realParameter[1623] /* theta[117] PARAM */))) * (((data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */)) * ((data->simulationInfo->realParameter[621] /* omega_c[117] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9162(DATA *data, threadData_t *threadData);


/*
equation index: 1871
type: SIMPLE_ASSIGN
vz[117] = 0.0
*/
void SpiralGalaxy_eqFunction_1871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1871};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1116]] /* vz[117] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9161(DATA *data, threadData_t *threadData);


/*
equation index: 1873
type: SIMPLE_ASSIGN
z[118] = -0.02112
*/
void SpiralGalaxy_eqFunction_1873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1873};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2617]] /* z[118] STATE(1,vz[118]) */) = -0.02112;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9174(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9175(DATA *data, threadData_t *threadData);


/*
equation index: 1876
type: SIMPLE_ASSIGN
y[118] = r_init[118] * sin(theta[118] - 0.00528)
*/
void SpiralGalaxy_eqFunction_1876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1876};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2117]] /* y[118] STATE(1,vy[118]) */) = ((data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */)) * (sin((data->simulationInfo->realParameter[1624] /* theta[118] PARAM */) - 0.00528));
  TRACE_POP
}

/*
equation index: 1877
type: SIMPLE_ASSIGN
x[118] = r_init[118] * cos(theta[118] - 0.00528)
*/
void SpiralGalaxy_eqFunction_1877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1617]] /* x[118] STATE(1,vx[118]) */) = ((data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */)) * (cos((data->simulationInfo->realParameter[1624] /* theta[118] PARAM */) - 0.00528));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9176(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9177(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9180(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9179(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9178(DATA *data, threadData_t *threadData);


/*
equation index: 1883
type: SIMPLE_ASSIGN
vx[118] = (-sin(theta[118])) * r_init[118] * omega_c[118]
*/
void SpiralGalaxy_eqFunction_1883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1883};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[117]] /* vx[118] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1624] /* theta[118] PARAM */)))) * (((data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */)) * ((data->simulationInfo->realParameter[622] /* omega_c[118] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9173(DATA *data, threadData_t *threadData);


/*
equation index: 1885
type: SIMPLE_ASSIGN
vy[118] = cos(theta[118]) * r_init[118] * omega_c[118]
*/
void SpiralGalaxy_eqFunction_1885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[617]] /* vy[118] STATE(1) */) = (cos((data->simulationInfo->realParameter[1624] /* theta[118] PARAM */))) * (((data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */)) * ((data->simulationInfo->realParameter[622] /* omega_c[118] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9172(DATA *data, threadData_t *threadData);


/*
equation index: 1887
type: SIMPLE_ASSIGN
vz[118] = 0.0
*/
void SpiralGalaxy_eqFunction_1887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1887};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1117]] /* vz[118] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9171(DATA *data, threadData_t *threadData);


/*
equation index: 1889
type: SIMPLE_ASSIGN
z[119] = -0.020960000000000003
*/
void SpiralGalaxy_eqFunction_1889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1889};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2618]] /* z[119] STATE(1,vz[119]) */) = -0.020960000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9184(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9185(DATA *data, threadData_t *threadData);


/*
equation index: 1892
type: SIMPLE_ASSIGN
y[119] = r_init[119] * sin(theta[119] - 0.005240000000000001)
*/
void SpiralGalaxy_eqFunction_1892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1892};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2118]] /* y[119] STATE(1,vy[119]) */) = ((data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */)) * (sin((data->simulationInfo->realParameter[1625] /* theta[119] PARAM */) - 0.005240000000000001));
  TRACE_POP
}

/*
equation index: 1893
type: SIMPLE_ASSIGN
x[119] = r_init[119] * cos(theta[119] - 0.005240000000000001)
*/
void SpiralGalaxy_eqFunction_1893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1618]] /* x[119] STATE(1,vx[119]) */) = ((data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */)) * (cos((data->simulationInfo->realParameter[1625] /* theta[119] PARAM */) - 0.005240000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9186(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9187(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9190(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9189(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9188(DATA *data, threadData_t *threadData);


/*
equation index: 1899
type: SIMPLE_ASSIGN
vx[119] = (-sin(theta[119])) * r_init[119] * omega_c[119]
*/
void SpiralGalaxy_eqFunction_1899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1899};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[118]] /* vx[119] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1625] /* theta[119] PARAM */)))) * (((data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */)) * ((data->simulationInfo->realParameter[623] /* omega_c[119] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9183(DATA *data, threadData_t *threadData);


/*
equation index: 1901
type: SIMPLE_ASSIGN
vy[119] = cos(theta[119]) * r_init[119] * omega_c[119]
*/
void SpiralGalaxy_eqFunction_1901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[618]] /* vy[119] STATE(1) */) = (cos((data->simulationInfo->realParameter[1625] /* theta[119] PARAM */))) * (((data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */)) * ((data->simulationInfo->realParameter[623] /* omega_c[119] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9182(DATA *data, threadData_t *threadData);


/*
equation index: 1903
type: SIMPLE_ASSIGN
vz[119] = 0.0
*/
void SpiralGalaxy_eqFunction_1903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1903};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1118]] /* vz[119] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9181(DATA *data, threadData_t *threadData);


/*
equation index: 1905
type: SIMPLE_ASSIGN
z[120] = -0.020800000000000006
*/
void SpiralGalaxy_eqFunction_1905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1905};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2619]] /* z[120] STATE(1,vz[120]) */) = -0.020800000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9194(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9195(DATA *data, threadData_t *threadData);


/*
equation index: 1908
type: SIMPLE_ASSIGN
y[120] = r_init[120] * sin(theta[120] - 0.005200000000000001)
*/
void SpiralGalaxy_eqFunction_1908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1908};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2119]] /* y[120] STATE(1,vy[120]) */) = ((data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */)) * (sin((data->simulationInfo->realParameter[1626] /* theta[120] PARAM */) - 0.005200000000000001));
  TRACE_POP
}

/*
equation index: 1909
type: SIMPLE_ASSIGN
x[120] = r_init[120] * cos(theta[120] - 0.005200000000000001)
*/
void SpiralGalaxy_eqFunction_1909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1619]] /* x[120] STATE(1,vx[120]) */) = ((data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */)) * (cos((data->simulationInfo->realParameter[1626] /* theta[120] PARAM */) - 0.005200000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9196(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9197(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9200(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9199(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9198(DATA *data, threadData_t *threadData);


/*
equation index: 1915
type: SIMPLE_ASSIGN
vx[120] = (-sin(theta[120])) * r_init[120] * omega_c[120]
*/
void SpiralGalaxy_eqFunction_1915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1915};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[119]] /* vx[120] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1626] /* theta[120] PARAM */)))) * (((data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */)) * ((data->simulationInfo->realParameter[624] /* omega_c[120] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9193(DATA *data, threadData_t *threadData);


/*
equation index: 1917
type: SIMPLE_ASSIGN
vy[120] = cos(theta[120]) * r_init[120] * omega_c[120]
*/
void SpiralGalaxy_eqFunction_1917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[619]] /* vy[120] STATE(1) */) = (cos((data->simulationInfo->realParameter[1626] /* theta[120] PARAM */))) * (((data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */)) * ((data->simulationInfo->realParameter[624] /* omega_c[120] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9192(DATA *data, threadData_t *threadData);


/*
equation index: 1919
type: SIMPLE_ASSIGN
vz[120] = 0.0
*/
void SpiralGalaxy_eqFunction_1919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1919};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1119]] /* vz[120] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9191(DATA *data, threadData_t *threadData);


/*
equation index: 1921
type: SIMPLE_ASSIGN
z[121] = -0.020640000000000002
*/
void SpiralGalaxy_eqFunction_1921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1921};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2620]] /* z[121] STATE(1,vz[121]) */) = -0.020640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9204(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9205(DATA *data, threadData_t *threadData);


/*
equation index: 1924
type: SIMPLE_ASSIGN
y[121] = r_init[121] * sin(theta[121] - 0.0051600000000000005)
*/
void SpiralGalaxy_eqFunction_1924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1924};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2120]] /* y[121] STATE(1,vy[121]) */) = ((data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */)) * (sin((data->simulationInfo->realParameter[1627] /* theta[121] PARAM */) - 0.0051600000000000005));
  TRACE_POP
}

/*
equation index: 1925
type: SIMPLE_ASSIGN
x[121] = r_init[121] * cos(theta[121] - 0.0051600000000000005)
*/
void SpiralGalaxy_eqFunction_1925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1620]] /* x[121] STATE(1,vx[121]) */) = ((data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */)) * (cos((data->simulationInfo->realParameter[1627] /* theta[121] PARAM */) - 0.0051600000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9206(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9207(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9210(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9209(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9208(DATA *data, threadData_t *threadData);


/*
equation index: 1931
type: SIMPLE_ASSIGN
vx[121] = (-sin(theta[121])) * r_init[121] * omega_c[121]
*/
void SpiralGalaxy_eqFunction_1931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1931};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[120]] /* vx[121] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1627] /* theta[121] PARAM */)))) * (((data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */)) * ((data->simulationInfo->realParameter[625] /* omega_c[121] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9203(DATA *data, threadData_t *threadData);


/*
equation index: 1933
type: SIMPLE_ASSIGN
vy[121] = cos(theta[121]) * r_init[121] * omega_c[121]
*/
void SpiralGalaxy_eqFunction_1933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[620]] /* vy[121] STATE(1) */) = (cos((data->simulationInfo->realParameter[1627] /* theta[121] PARAM */))) * (((data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */)) * ((data->simulationInfo->realParameter[625] /* omega_c[121] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9202(DATA *data, threadData_t *threadData);


/*
equation index: 1935
type: SIMPLE_ASSIGN
vz[121] = 0.0
*/
void SpiralGalaxy_eqFunction_1935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1935};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1120]] /* vz[121] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9201(DATA *data, threadData_t *threadData);


/*
equation index: 1937
type: SIMPLE_ASSIGN
z[122] = -0.02048
*/
void SpiralGalaxy_eqFunction_1937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1937};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2621]] /* z[122] STATE(1,vz[122]) */) = -0.02048;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9214(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9215(DATA *data, threadData_t *threadData);


/*
equation index: 1940
type: SIMPLE_ASSIGN
y[122] = r_init[122] * sin(theta[122] - 0.00512)
*/
void SpiralGalaxy_eqFunction_1940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1940};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2121]] /* y[122] STATE(1,vy[122]) */) = ((data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */)) * (sin((data->simulationInfo->realParameter[1628] /* theta[122] PARAM */) - 0.00512));
  TRACE_POP
}

/*
equation index: 1941
type: SIMPLE_ASSIGN
x[122] = r_init[122] * cos(theta[122] - 0.00512)
*/
void SpiralGalaxy_eqFunction_1941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1621]] /* x[122] STATE(1,vx[122]) */) = ((data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */)) * (cos((data->simulationInfo->realParameter[1628] /* theta[122] PARAM */) - 0.00512));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9216(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9217(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9220(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9219(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9218(DATA *data, threadData_t *threadData);


/*
equation index: 1947
type: SIMPLE_ASSIGN
vx[122] = (-sin(theta[122])) * r_init[122] * omega_c[122]
*/
void SpiralGalaxy_eqFunction_1947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1947};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[121]] /* vx[122] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1628] /* theta[122] PARAM */)))) * (((data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */)) * ((data->simulationInfo->realParameter[626] /* omega_c[122] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9213(DATA *data, threadData_t *threadData);


/*
equation index: 1949
type: SIMPLE_ASSIGN
vy[122] = cos(theta[122]) * r_init[122] * omega_c[122]
*/
void SpiralGalaxy_eqFunction_1949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[621]] /* vy[122] STATE(1) */) = (cos((data->simulationInfo->realParameter[1628] /* theta[122] PARAM */))) * (((data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */)) * ((data->simulationInfo->realParameter[626] /* omega_c[122] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9212(DATA *data, threadData_t *threadData);


/*
equation index: 1951
type: SIMPLE_ASSIGN
vz[122] = 0.0
*/
void SpiralGalaxy_eqFunction_1951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1951};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1121]] /* vz[122] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9211(DATA *data, threadData_t *threadData);


/*
equation index: 1953
type: SIMPLE_ASSIGN
z[123] = -0.02032
*/
void SpiralGalaxy_eqFunction_1953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1953};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2622]] /* z[123] STATE(1,vz[123]) */) = -0.02032;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9224(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9225(DATA *data, threadData_t *threadData);


/*
equation index: 1956
type: SIMPLE_ASSIGN
y[123] = r_init[123] * sin(theta[123] - 0.00508)
*/
void SpiralGalaxy_eqFunction_1956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1956};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2122]] /* y[123] STATE(1,vy[123]) */) = ((data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */)) * (sin((data->simulationInfo->realParameter[1629] /* theta[123] PARAM */) - 0.00508));
  TRACE_POP
}

/*
equation index: 1957
type: SIMPLE_ASSIGN
x[123] = r_init[123] * cos(theta[123] - 0.00508)
*/
void SpiralGalaxy_eqFunction_1957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1622]] /* x[123] STATE(1,vx[123]) */) = ((data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */)) * (cos((data->simulationInfo->realParameter[1629] /* theta[123] PARAM */) - 0.00508));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9226(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9227(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9230(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9229(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9228(DATA *data, threadData_t *threadData);


/*
equation index: 1963
type: SIMPLE_ASSIGN
vx[123] = (-sin(theta[123])) * r_init[123] * omega_c[123]
*/
void SpiralGalaxy_eqFunction_1963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1963};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[122]] /* vx[123] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1629] /* theta[123] PARAM */)))) * (((data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */)) * ((data->simulationInfo->realParameter[627] /* omega_c[123] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9223(DATA *data, threadData_t *threadData);


/*
equation index: 1965
type: SIMPLE_ASSIGN
vy[123] = cos(theta[123]) * r_init[123] * omega_c[123]
*/
void SpiralGalaxy_eqFunction_1965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[622]] /* vy[123] STATE(1) */) = (cos((data->simulationInfo->realParameter[1629] /* theta[123] PARAM */))) * (((data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */)) * ((data->simulationInfo->realParameter[627] /* omega_c[123] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9222(DATA *data, threadData_t *threadData);


/*
equation index: 1967
type: SIMPLE_ASSIGN
vz[123] = 0.0
*/
void SpiralGalaxy_eqFunction_1967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1967};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1122]] /* vz[123] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9221(DATA *data, threadData_t *threadData);


/*
equation index: 1969
type: SIMPLE_ASSIGN
z[124] = -0.02016
*/
void SpiralGalaxy_eqFunction_1969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1969};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2623]] /* z[124] STATE(1,vz[124]) */) = -0.02016;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9234(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9235(DATA *data, threadData_t *threadData);


/*
equation index: 1972
type: SIMPLE_ASSIGN
y[124] = r_init[124] * sin(theta[124] - 0.00504)
*/
void SpiralGalaxy_eqFunction_1972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1972};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2123]] /* y[124] STATE(1,vy[124]) */) = ((data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */)) * (sin((data->simulationInfo->realParameter[1630] /* theta[124] PARAM */) - 0.00504));
  TRACE_POP
}

/*
equation index: 1973
type: SIMPLE_ASSIGN
x[124] = r_init[124] * cos(theta[124] - 0.00504)
*/
void SpiralGalaxy_eqFunction_1973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1623]] /* x[124] STATE(1,vx[124]) */) = ((data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */)) * (cos((data->simulationInfo->realParameter[1630] /* theta[124] PARAM */) - 0.00504));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9236(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9237(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9240(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9239(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9238(DATA *data, threadData_t *threadData);


/*
equation index: 1979
type: SIMPLE_ASSIGN
vx[124] = (-sin(theta[124])) * r_init[124] * omega_c[124]
*/
void SpiralGalaxy_eqFunction_1979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1979};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[123]] /* vx[124] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1630] /* theta[124] PARAM */)))) * (((data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */)) * ((data->simulationInfo->realParameter[628] /* omega_c[124] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9233(DATA *data, threadData_t *threadData);


/*
equation index: 1981
type: SIMPLE_ASSIGN
vy[124] = cos(theta[124]) * r_init[124] * omega_c[124]
*/
void SpiralGalaxy_eqFunction_1981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[623]] /* vy[124] STATE(1) */) = (cos((data->simulationInfo->realParameter[1630] /* theta[124] PARAM */))) * (((data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */)) * ((data->simulationInfo->realParameter[628] /* omega_c[124] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9232(DATA *data, threadData_t *threadData);


/*
equation index: 1983
type: SIMPLE_ASSIGN
vz[124] = 0.0
*/
void SpiralGalaxy_eqFunction_1983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1983};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1123]] /* vz[124] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9231(DATA *data, threadData_t *threadData);


/*
equation index: 1985
type: SIMPLE_ASSIGN
z[125] = -0.02
*/
void SpiralGalaxy_eqFunction_1985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1985};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2624]] /* z[125] STATE(1,vz[125]) */) = -0.02;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9244(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9245(DATA *data, threadData_t *threadData);


/*
equation index: 1988
type: SIMPLE_ASSIGN
y[125] = r_init[125] * sin(theta[125] - 0.005)
*/
void SpiralGalaxy_eqFunction_1988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1988};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2124]] /* y[125] STATE(1,vy[125]) */) = ((data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */)) * (sin((data->simulationInfo->realParameter[1631] /* theta[125] PARAM */) - 0.005));
  TRACE_POP
}

/*
equation index: 1989
type: SIMPLE_ASSIGN
x[125] = r_init[125] * cos(theta[125] - 0.005)
*/
void SpiralGalaxy_eqFunction_1989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1624]] /* x[125] STATE(1,vx[125]) */) = ((data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */)) * (cos((data->simulationInfo->realParameter[1631] /* theta[125] PARAM */) - 0.005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9246(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9247(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9250(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9249(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9248(DATA *data, threadData_t *threadData);


/*
equation index: 1995
type: SIMPLE_ASSIGN
vx[125] = (-sin(theta[125])) * r_init[125] * omega_c[125]
*/
void SpiralGalaxy_eqFunction_1995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1995};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[124]] /* vx[125] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1631] /* theta[125] PARAM */)))) * (((data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */)) * ((data->simulationInfo->realParameter[629] /* omega_c[125] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9243(DATA *data, threadData_t *threadData);


/*
equation index: 1997
type: SIMPLE_ASSIGN
vy[125] = cos(theta[125]) * r_init[125] * omega_c[125]
*/
void SpiralGalaxy_eqFunction_1997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[624]] /* vy[125] STATE(1) */) = (cos((data->simulationInfo->realParameter[1631] /* theta[125] PARAM */))) * (((data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */)) * ((data->simulationInfo->realParameter[629] /* omega_c[125] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9242(DATA *data, threadData_t *threadData);


/*
equation index: 1999
type: SIMPLE_ASSIGN
vz[125] = 0.0
*/
void SpiralGalaxy_eqFunction_1999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1999};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1124]] /* vz[125] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9241(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_3(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_1501(data, threadData);
  SpiralGalaxy_eqFunction_8932(data, threadData);
  SpiralGalaxy_eqFunction_1503(data, threadData);
  SpiralGalaxy_eqFunction_8931(data, threadData);
  SpiralGalaxy_eqFunction_1505(data, threadData);
  SpiralGalaxy_eqFunction_8944(data, threadData);
  SpiralGalaxy_eqFunction_8945(data, threadData);
  SpiralGalaxy_eqFunction_1508(data, threadData);
  SpiralGalaxy_eqFunction_1509(data, threadData);
  SpiralGalaxy_eqFunction_8946(data, threadData);
  SpiralGalaxy_eqFunction_8947(data, threadData);
  SpiralGalaxy_eqFunction_8950(data, threadData);
  SpiralGalaxy_eqFunction_8949(data, threadData);
  SpiralGalaxy_eqFunction_8948(data, threadData);
  SpiralGalaxy_eqFunction_1515(data, threadData);
  SpiralGalaxy_eqFunction_8943(data, threadData);
  SpiralGalaxy_eqFunction_1517(data, threadData);
  SpiralGalaxy_eqFunction_8942(data, threadData);
  SpiralGalaxy_eqFunction_1519(data, threadData);
  SpiralGalaxy_eqFunction_8941(data, threadData);
  SpiralGalaxy_eqFunction_1521(data, threadData);
  SpiralGalaxy_eqFunction_8954(data, threadData);
  SpiralGalaxy_eqFunction_8955(data, threadData);
  SpiralGalaxy_eqFunction_1524(data, threadData);
  SpiralGalaxy_eqFunction_1525(data, threadData);
  SpiralGalaxy_eqFunction_8956(data, threadData);
  SpiralGalaxy_eqFunction_8957(data, threadData);
  SpiralGalaxy_eqFunction_8960(data, threadData);
  SpiralGalaxy_eqFunction_8959(data, threadData);
  SpiralGalaxy_eqFunction_8958(data, threadData);
  SpiralGalaxy_eqFunction_1531(data, threadData);
  SpiralGalaxy_eqFunction_8953(data, threadData);
  SpiralGalaxy_eqFunction_1533(data, threadData);
  SpiralGalaxy_eqFunction_8952(data, threadData);
  SpiralGalaxy_eqFunction_1535(data, threadData);
  SpiralGalaxy_eqFunction_8951(data, threadData);
  SpiralGalaxy_eqFunction_1537(data, threadData);
  SpiralGalaxy_eqFunction_8964(data, threadData);
  SpiralGalaxy_eqFunction_8965(data, threadData);
  SpiralGalaxy_eqFunction_1540(data, threadData);
  SpiralGalaxy_eqFunction_1541(data, threadData);
  SpiralGalaxy_eqFunction_8966(data, threadData);
  SpiralGalaxy_eqFunction_8967(data, threadData);
  SpiralGalaxy_eqFunction_8970(data, threadData);
  SpiralGalaxy_eqFunction_8969(data, threadData);
  SpiralGalaxy_eqFunction_8968(data, threadData);
  SpiralGalaxy_eqFunction_1547(data, threadData);
  SpiralGalaxy_eqFunction_8963(data, threadData);
  SpiralGalaxy_eqFunction_1549(data, threadData);
  SpiralGalaxy_eqFunction_8962(data, threadData);
  SpiralGalaxy_eqFunction_1551(data, threadData);
  SpiralGalaxy_eqFunction_8961(data, threadData);
  SpiralGalaxy_eqFunction_1553(data, threadData);
  SpiralGalaxy_eqFunction_8974(data, threadData);
  SpiralGalaxy_eqFunction_8975(data, threadData);
  SpiralGalaxy_eqFunction_1556(data, threadData);
  SpiralGalaxy_eqFunction_1557(data, threadData);
  SpiralGalaxy_eqFunction_8976(data, threadData);
  SpiralGalaxy_eqFunction_8977(data, threadData);
  SpiralGalaxy_eqFunction_8980(data, threadData);
  SpiralGalaxy_eqFunction_8979(data, threadData);
  SpiralGalaxy_eqFunction_8978(data, threadData);
  SpiralGalaxy_eqFunction_1563(data, threadData);
  SpiralGalaxy_eqFunction_8973(data, threadData);
  SpiralGalaxy_eqFunction_1565(data, threadData);
  SpiralGalaxy_eqFunction_8972(data, threadData);
  SpiralGalaxy_eqFunction_1567(data, threadData);
  SpiralGalaxy_eqFunction_8971(data, threadData);
  SpiralGalaxy_eqFunction_1569(data, threadData);
  SpiralGalaxy_eqFunction_8984(data, threadData);
  SpiralGalaxy_eqFunction_8985(data, threadData);
  SpiralGalaxy_eqFunction_1572(data, threadData);
  SpiralGalaxy_eqFunction_1573(data, threadData);
  SpiralGalaxy_eqFunction_8986(data, threadData);
  SpiralGalaxy_eqFunction_8987(data, threadData);
  SpiralGalaxy_eqFunction_8990(data, threadData);
  SpiralGalaxy_eqFunction_8989(data, threadData);
  SpiralGalaxy_eqFunction_8988(data, threadData);
  SpiralGalaxy_eqFunction_1579(data, threadData);
  SpiralGalaxy_eqFunction_8983(data, threadData);
  SpiralGalaxy_eqFunction_1581(data, threadData);
  SpiralGalaxy_eqFunction_8982(data, threadData);
  SpiralGalaxy_eqFunction_1583(data, threadData);
  SpiralGalaxy_eqFunction_8981(data, threadData);
  SpiralGalaxy_eqFunction_1585(data, threadData);
  SpiralGalaxy_eqFunction_8994(data, threadData);
  SpiralGalaxy_eqFunction_8995(data, threadData);
  SpiralGalaxy_eqFunction_1588(data, threadData);
  SpiralGalaxy_eqFunction_1589(data, threadData);
  SpiralGalaxy_eqFunction_8996(data, threadData);
  SpiralGalaxy_eqFunction_8997(data, threadData);
  SpiralGalaxy_eqFunction_9000(data, threadData);
  SpiralGalaxy_eqFunction_8999(data, threadData);
  SpiralGalaxy_eqFunction_8998(data, threadData);
  SpiralGalaxy_eqFunction_1595(data, threadData);
  SpiralGalaxy_eqFunction_8993(data, threadData);
  SpiralGalaxy_eqFunction_1597(data, threadData);
  SpiralGalaxy_eqFunction_8992(data, threadData);
  SpiralGalaxy_eqFunction_1599(data, threadData);
  SpiralGalaxy_eqFunction_8991(data, threadData);
  SpiralGalaxy_eqFunction_1601(data, threadData);
  SpiralGalaxy_eqFunction_9004(data, threadData);
  SpiralGalaxy_eqFunction_9005(data, threadData);
  SpiralGalaxy_eqFunction_1604(data, threadData);
  SpiralGalaxy_eqFunction_1605(data, threadData);
  SpiralGalaxy_eqFunction_9006(data, threadData);
  SpiralGalaxy_eqFunction_9007(data, threadData);
  SpiralGalaxy_eqFunction_9010(data, threadData);
  SpiralGalaxy_eqFunction_9009(data, threadData);
  SpiralGalaxy_eqFunction_9008(data, threadData);
  SpiralGalaxy_eqFunction_1611(data, threadData);
  SpiralGalaxy_eqFunction_9003(data, threadData);
  SpiralGalaxy_eqFunction_1613(data, threadData);
  SpiralGalaxy_eqFunction_9002(data, threadData);
  SpiralGalaxy_eqFunction_1615(data, threadData);
  SpiralGalaxy_eqFunction_9001(data, threadData);
  SpiralGalaxy_eqFunction_1617(data, threadData);
  SpiralGalaxy_eqFunction_9014(data, threadData);
  SpiralGalaxy_eqFunction_9015(data, threadData);
  SpiralGalaxy_eqFunction_1620(data, threadData);
  SpiralGalaxy_eqFunction_1621(data, threadData);
  SpiralGalaxy_eqFunction_9016(data, threadData);
  SpiralGalaxy_eqFunction_9017(data, threadData);
  SpiralGalaxy_eqFunction_9020(data, threadData);
  SpiralGalaxy_eqFunction_9019(data, threadData);
  SpiralGalaxy_eqFunction_9018(data, threadData);
  SpiralGalaxy_eqFunction_1627(data, threadData);
  SpiralGalaxy_eqFunction_9013(data, threadData);
  SpiralGalaxy_eqFunction_1629(data, threadData);
  SpiralGalaxy_eqFunction_9012(data, threadData);
  SpiralGalaxy_eqFunction_1631(data, threadData);
  SpiralGalaxy_eqFunction_9011(data, threadData);
  SpiralGalaxy_eqFunction_1633(data, threadData);
  SpiralGalaxy_eqFunction_9024(data, threadData);
  SpiralGalaxy_eqFunction_9025(data, threadData);
  SpiralGalaxy_eqFunction_1636(data, threadData);
  SpiralGalaxy_eqFunction_1637(data, threadData);
  SpiralGalaxy_eqFunction_9026(data, threadData);
  SpiralGalaxy_eqFunction_9027(data, threadData);
  SpiralGalaxy_eqFunction_9030(data, threadData);
  SpiralGalaxy_eqFunction_9029(data, threadData);
  SpiralGalaxy_eqFunction_9028(data, threadData);
  SpiralGalaxy_eqFunction_1643(data, threadData);
  SpiralGalaxy_eqFunction_9023(data, threadData);
  SpiralGalaxy_eqFunction_1645(data, threadData);
  SpiralGalaxy_eqFunction_9022(data, threadData);
  SpiralGalaxy_eqFunction_1647(data, threadData);
  SpiralGalaxy_eqFunction_9021(data, threadData);
  SpiralGalaxy_eqFunction_1649(data, threadData);
  SpiralGalaxy_eqFunction_9034(data, threadData);
  SpiralGalaxy_eqFunction_9035(data, threadData);
  SpiralGalaxy_eqFunction_1652(data, threadData);
  SpiralGalaxy_eqFunction_1653(data, threadData);
  SpiralGalaxy_eqFunction_9036(data, threadData);
  SpiralGalaxy_eqFunction_9037(data, threadData);
  SpiralGalaxy_eqFunction_9040(data, threadData);
  SpiralGalaxy_eqFunction_9039(data, threadData);
  SpiralGalaxy_eqFunction_9038(data, threadData);
  SpiralGalaxy_eqFunction_1659(data, threadData);
  SpiralGalaxy_eqFunction_9033(data, threadData);
  SpiralGalaxy_eqFunction_1661(data, threadData);
  SpiralGalaxy_eqFunction_9032(data, threadData);
  SpiralGalaxy_eqFunction_1663(data, threadData);
  SpiralGalaxy_eqFunction_9031(data, threadData);
  SpiralGalaxy_eqFunction_1665(data, threadData);
  SpiralGalaxy_eqFunction_9044(data, threadData);
  SpiralGalaxy_eqFunction_9045(data, threadData);
  SpiralGalaxy_eqFunction_1668(data, threadData);
  SpiralGalaxy_eqFunction_1669(data, threadData);
  SpiralGalaxy_eqFunction_9046(data, threadData);
  SpiralGalaxy_eqFunction_9047(data, threadData);
  SpiralGalaxy_eqFunction_9050(data, threadData);
  SpiralGalaxy_eqFunction_9049(data, threadData);
  SpiralGalaxy_eqFunction_9048(data, threadData);
  SpiralGalaxy_eqFunction_1675(data, threadData);
  SpiralGalaxy_eqFunction_9043(data, threadData);
  SpiralGalaxy_eqFunction_1677(data, threadData);
  SpiralGalaxy_eqFunction_9042(data, threadData);
  SpiralGalaxy_eqFunction_1679(data, threadData);
  SpiralGalaxy_eqFunction_9041(data, threadData);
  SpiralGalaxy_eqFunction_1681(data, threadData);
  SpiralGalaxy_eqFunction_9054(data, threadData);
  SpiralGalaxy_eqFunction_9055(data, threadData);
  SpiralGalaxy_eqFunction_1684(data, threadData);
  SpiralGalaxy_eqFunction_1685(data, threadData);
  SpiralGalaxy_eqFunction_9056(data, threadData);
  SpiralGalaxy_eqFunction_9057(data, threadData);
  SpiralGalaxy_eqFunction_9060(data, threadData);
  SpiralGalaxy_eqFunction_9059(data, threadData);
  SpiralGalaxy_eqFunction_9058(data, threadData);
  SpiralGalaxy_eqFunction_1691(data, threadData);
  SpiralGalaxy_eqFunction_9053(data, threadData);
  SpiralGalaxy_eqFunction_1693(data, threadData);
  SpiralGalaxy_eqFunction_9052(data, threadData);
  SpiralGalaxy_eqFunction_1695(data, threadData);
  SpiralGalaxy_eqFunction_9051(data, threadData);
  SpiralGalaxy_eqFunction_1697(data, threadData);
  SpiralGalaxy_eqFunction_9064(data, threadData);
  SpiralGalaxy_eqFunction_9065(data, threadData);
  SpiralGalaxy_eqFunction_1700(data, threadData);
  SpiralGalaxy_eqFunction_1701(data, threadData);
  SpiralGalaxy_eqFunction_9066(data, threadData);
  SpiralGalaxy_eqFunction_9067(data, threadData);
  SpiralGalaxy_eqFunction_9070(data, threadData);
  SpiralGalaxy_eqFunction_9069(data, threadData);
  SpiralGalaxy_eqFunction_9068(data, threadData);
  SpiralGalaxy_eqFunction_1707(data, threadData);
  SpiralGalaxy_eqFunction_9063(data, threadData);
  SpiralGalaxy_eqFunction_1709(data, threadData);
  SpiralGalaxy_eqFunction_9062(data, threadData);
  SpiralGalaxy_eqFunction_1711(data, threadData);
  SpiralGalaxy_eqFunction_9061(data, threadData);
  SpiralGalaxy_eqFunction_1713(data, threadData);
  SpiralGalaxy_eqFunction_9074(data, threadData);
  SpiralGalaxy_eqFunction_9075(data, threadData);
  SpiralGalaxy_eqFunction_1716(data, threadData);
  SpiralGalaxy_eqFunction_1717(data, threadData);
  SpiralGalaxy_eqFunction_9076(data, threadData);
  SpiralGalaxy_eqFunction_9077(data, threadData);
  SpiralGalaxy_eqFunction_9080(data, threadData);
  SpiralGalaxy_eqFunction_9079(data, threadData);
  SpiralGalaxy_eqFunction_9078(data, threadData);
  SpiralGalaxy_eqFunction_1723(data, threadData);
  SpiralGalaxy_eqFunction_9073(data, threadData);
  SpiralGalaxy_eqFunction_1725(data, threadData);
  SpiralGalaxy_eqFunction_9072(data, threadData);
  SpiralGalaxy_eqFunction_1727(data, threadData);
  SpiralGalaxy_eqFunction_9071(data, threadData);
  SpiralGalaxy_eqFunction_1729(data, threadData);
  SpiralGalaxy_eqFunction_9084(data, threadData);
  SpiralGalaxy_eqFunction_9085(data, threadData);
  SpiralGalaxy_eqFunction_1732(data, threadData);
  SpiralGalaxy_eqFunction_1733(data, threadData);
  SpiralGalaxy_eqFunction_9086(data, threadData);
  SpiralGalaxy_eqFunction_9087(data, threadData);
  SpiralGalaxy_eqFunction_9090(data, threadData);
  SpiralGalaxy_eqFunction_9089(data, threadData);
  SpiralGalaxy_eqFunction_9088(data, threadData);
  SpiralGalaxy_eqFunction_1739(data, threadData);
  SpiralGalaxy_eqFunction_9083(data, threadData);
  SpiralGalaxy_eqFunction_1741(data, threadData);
  SpiralGalaxy_eqFunction_9082(data, threadData);
  SpiralGalaxy_eqFunction_1743(data, threadData);
  SpiralGalaxy_eqFunction_9081(data, threadData);
  SpiralGalaxy_eqFunction_1745(data, threadData);
  SpiralGalaxy_eqFunction_9094(data, threadData);
  SpiralGalaxy_eqFunction_9095(data, threadData);
  SpiralGalaxy_eqFunction_1748(data, threadData);
  SpiralGalaxy_eqFunction_1749(data, threadData);
  SpiralGalaxy_eqFunction_9096(data, threadData);
  SpiralGalaxy_eqFunction_9097(data, threadData);
  SpiralGalaxy_eqFunction_9100(data, threadData);
  SpiralGalaxy_eqFunction_9099(data, threadData);
  SpiralGalaxy_eqFunction_9098(data, threadData);
  SpiralGalaxy_eqFunction_1755(data, threadData);
  SpiralGalaxy_eqFunction_9093(data, threadData);
  SpiralGalaxy_eqFunction_1757(data, threadData);
  SpiralGalaxy_eqFunction_9092(data, threadData);
  SpiralGalaxy_eqFunction_1759(data, threadData);
  SpiralGalaxy_eqFunction_9091(data, threadData);
  SpiralGalaxy_eqFunction_1761(data, threadData);
  SpiralGalaxy_eqFunction_9104(data, threadData);
  SpiralGalaxy_eqFunction_9105(data, threadData);
  SpiralGalaxy_eqFunction_1764(data, threadData);
  SpiralGalaxy_eqFunction_1765(data, threadData);
  SpiralGalaxy_eqFunction_9106(data, threadData);
  SpiralGalaxy_eqFunction_9107(data, threadData);
  SpiralGalaxy_eqFunction_9110(data, threadData);
  SpiralGalaxy_eqFunction_9109(data, threadData);
  SpiralGalaxy_eqFunction_9108(data, threadData);
  SpiralGalaxy_eqFunction_1771(data, threadData);
  SpiralGalaxy_eqFunction_9103(data, threadData);
  SpiralGalaxy_eqFunction_1773(data, threadData);
  SpiralGalaxy_eqFunction_9102(data, threadData);
  SpiralGalaxy_eqFunction_1775(data, threadData);
  SpiralGalaxy_eqFunction_9101(data, threadData);
  SpiralGalaxy_eqFunction_1777(data, threadData);
  SpiralGalaxy_eqFunction_9114(data, threadData);
  SpiralGalaxy_eqFunction_9115(data, threadData);
  SpiralGalaxy_eqFunction_1780(data, threadData);
  SpiralGalaxy_eqFunction_1781(data, threadData);
  SpiralGalaxy_eqFunction_9116(data, threadData);
  SpiralGalaxy_eqFunction_9117(data, threadData);
  SpiralGalaxy_eqFunction_9120(data, threadData);
  SpiralGalaxy_eqFunction_9119(data, threadData);
  SpiralGalaxy_eqFunction_9118(data, threadData);
  SpiralGalaxy_eqFunction_1787(data, threadData);
  SpiralGalaxy_eqFunction_9113(data, threadData);
  SpiralGalaxy_eqFunction_1789(data, threadData);
  SpiralGalaxy_eqFunction_9112(data, threadData);
  SpiralGalaxy_eqFunction_1791(data, threadData);
  SpiralGalaxy_eqFunction_9111(data, threadData);
  SpiralGalaxy_eqFunction_1793(data, threadData);
  SpiralGalaxy_eqFunction_9124(data, threadData);
  SpiralGalaxy_eqFunction_9125(data, threadData);
  SpiralGalaxy_eqFunction_1796(data, threadData);
  SpiralGalaxy_eqFunction_1797(data, threadData);
  SpiralGalaxy_eqFunction_9126(data, threadData);
  SpiralGalaxy_eqFunction_9127(data, threadData);
  SpiralGalaxy_eqFunction_9130(data, threadData);
  SpiralGalaxy_eqFunction_9129(data, threadData);
  SpiralGalaxy_eqFunction_9128(data, threadData);
  SpiralGalaxy_eqFunction_1803(data, threadData);
  SpiralGalaxy_eqFunction_9123(data, threadData);
  SpiralGalaxy_eqFunction_1805(data, threadData);
  SpiralGalaxy_eqFunction_9122(data, threadData);
  SpiralGalaxy_eqFunction_1807(data, threadData);
  SpiralGalaxy_eqFunction_9121(data, threadData);
  SpiralGalaxy_eqFunction_1809(data, threadData);
  SpiralGalaxy_eqFunction_9134(data, threadData);
  SpiralGalaxy_eqFunction_9135(data, threadData);
  SpiralGalaxy_eqFunction_1812(data, threadData);
  SpiralGalaxy_eqFunction_1813(data, threadData);
  SpiralGalaxy_eqFunction_9136(data, threadData);
  SpiralGalaxy_eqFunction_9137(data, threadData);
  SpiralGalaxy_eqFunction_9140(data, threadData);
  SpiralGalaxy_eqFunction_9139(data, threadData);
  SpiralGalaxy_eqFunction_9138(data, threadData);
  SpiralGalaxy_eqFunction_1819(data, threadData);
  SpiralGalaxy_eqFunction_9133(data, threadData);
  SpiralGalaxy_eqFunction_1821(data, threadData);
  SpiralGalaxy_eqFunction_9132(data, threadData);
  SpiralGalaxy_eqFunction_1823(data, threadData);
  SpiralGalaxy_eqFunction_9131(data, threadData);
  SpiralGalaxy_eqFunction_1825(data, threadData);
  SpiralGalaxy_eqFunction_9144(data, threadData);
  SpiralGalaxy_eqFunction_9145(data, threadData);
  SpiralGalaxy_eqFunction_1828(data, threadData);
  SpiralGalaxy_eqFunction_1829(data, threadData);
  SpiralGalaxy_eqFunction_9146(data, threadData);
  SpiralGalaxy_eqFunction_9147(data, threadData);
  SpiralGalaxy_eqFunction_9150(data, threadData);
  SpiralGalaxy_eqFunction_9149(data, threadData);
  SpiralGalaxy_eqFunction_9148(data, threadData);
  SpiralGalaxy_eqFunction_1835(data, threadData);
  SpiralGalaxy_eqFunction_9143(data, threadData);
  SpiralGalaxy_eqFunction_1837(data, threadData);
  SpiralGalaxy_eqFunction_9142(data, threadData);
  SpiralGalaxy_eqFunction_1839(data, threadData);
  SpiralGalaxy_eqFunction_9141(data, threadData);
  SpiralGalaxy_eqFunction_1841(data, threadData);
  SpiralGalaxy_eqFunction_9154(data, threadData);
  SpiralGalaxy_eqFunction_9155(data, threadData);
  SpiralGalaxy_eqFunction_1844(data, threadData);
  SpiralGalaxy_eqFunction_1845(data, threadData);
  SpiralGalaxy_eqFunction_9156(data, threadData);
  SpiralGalaxy_eqFunction_9157(data, threadData);
  SpiralGalaxy_eqFunction_9160(data, threadData);
  SpiralGalaxy_eqFunction_9159(data, threadData);
  SpiralGalaxy_eqFunction_9158(data, threadData);
  SpiralGalaxy_eqFunction_1851(data, threadData);
  SpiralGalaxy_eqFunction_9153(data, threadData);
  SpiralGalaxy_eqFunction_1853(data, threadData);
  SpiralGalaxy_eqFunction_9152(data, threadData);
  SpiralGalaxy_eqFunction_1855(data, threadData);
  SpiralGalaxy_eqFunction_9151(data, threadData);
  SpiralGalaxy_eqFunction_1857(data, threadData);
  SpiralGalaxy_eqFunction_9164(data, threadData);
  SpiralGalaxy_eqFunction_9165(data, threadData);
  SpiralGalaxy_eqFunction_1860(data, threadData);
  SpiralGalaxy_eqFunction_1861(data, threadData);
  SpiralGalaxy_eqFunction_9166(data, threadData);
  SpiralGalaxy_eqFunction_9167(data, threadData);
  SpiralGalaxy_eqFunction_9170(data, threadData);
  SpiralGalaxy_eqFunction_9169(data, threadData);
  SpiralGalaxy_eqFunction_9168(data, threadData);
  SpiralGalaxy_eqFunction_1867(data, threadData);
  SpiralGalaxy_eqFunction_9163(data, threadData);
  SpiralGalaxy_eqFunction_1869(data, threadData);
  SpiralGalaxy_eqFunction_9162(data, threadData);
  SpiralGalaxy_eqFunction_1871(data, threadData);
  SpiralGalaxy_eqFunction_9161(data, threadData);
  SpiralGalaxy_eqFunction_1873(data, threadData);
  SpiralGalaxy_eqFunction_9174(data, threadData);
  SpiralGalaxy_eqFunction_9175(data, threadData);
  SpiralGalaxy_eqFunction_1876(data, threadData);
  SpiralGalaxy_eqFunction_1877(data, threadData);
  SpiralGalaxy_eqFunction_9176(data, threadData);
  SpiralGalaxy_eqFunction_9177(data, threadData);
  SpiralGalaxy_eqFunction_9180(data, threadData);
  SpiralGalaxy_eqFunction_9179(data, threadData);
  SpiralGalaxy_eqFunction_9178(data, threadData);
  SpiralGalaxy_eqFunction_1883(data, threadData);
  SpiralGalaxy_eqFunction_9173(data, threadData);
  SpiralGalaxy_eqFunction_1885(data, threadData);
  SpiralGalaxy_eqFunction_9172(data, threadData);
  SpiralGalaxy_eqFunction_1887(data, threadData);
  SpiralGalaxy_eqFunction_9171(data, threadData);
  SpiralGalaxy_eqFunction_1889(data, threadData);
  SpiralGalaxy_eqFunction_9184(data, threadData);
  SpiralGalaxy_eqFunction_9185(data, threadData);
  SpiralGalaxy_eqFunction_1892(data, threadData);
  SpiralGalaxy_eqFunction_1893(data, threadData);
  SpiralGalaxy_eqFunction_9186(data, threadData);
  SpiralGalaxy_eqFunction_9187(data, threadData);
  SpiralGalaxy_eqFunction_9190(data, threadData);
  SpiralGalaxy_eqFunction_9189(data, threadData);
  SpiralGalaxy_eqFunction_9188(data, threadData);
  SpiralGalaxy_eqFunction_1899(data, threadData);
  SpiralGalaxy_eqFunction_9183(data, threadData);
  SpiralGalaxy_eqFunction_1901(data, threadData);
  SpiralGalaxy_eqFunction_9182(data, threadData);
  SpiralGalaxy_eqFunction_1903(data, threadData);
  SpiralGalaxy_eqFunction_9181(data, threadData);
  SpiralGalaxy_eqFunction_1905(data, threadData);
  SpiralGalaxy_eqFunction_9194(data, threadData);
  SpiralGalaxy_eqFunction_9195(data, threadData);
  SpiralGalaxy_eqFunction_1908(data, threadData);
  SpiralGalaxy_eqFunction_1909(data, threadData);
  SpiralGalaxy_eqFunction_9196(data, threadData);
  SpiralGalaxy_eqFunction_9197(data, threadData);
  SpiralGalaxy_eqFunction_9200(data, threadData);
  SpiralGalaxy_eqFunction_9199(data, threadData);
  SpiralGalaxy_eqFunction_9198(data, threadData);
  SpiralGalaxy_eqFunction_1915(data, threadData);
  SpiralGalaxy_eqFunction_9193(data, threadData);
  SpiralGalaxy_eqFunction_1917(data, threadData);
  SpiralGalaxy_eqFunction_9192(data, threadData);
  SpiralGalaxy_eqFunction_1919(data, threadData);
  SpiralGalaxy_eqFunction_9191(data, threadData);
  SpiralGalaxy_eqFunction_1921(data, threadData);
  SpiralGalaxy_eqFunction_9204(data, threadData);
  SpiralGalaxy_eqFunction_9205(data, threadData);
  SpiralGalaxy_eqFunction_1924(data, threadData);
  SpiralGalaxy_eqFunction_1925(data, threadData);
  SpiralGalaxy_eqFunction_9206(data, threadData);
  SpiralGalaxy_eqFunction_9207(data, threadData);
  SpiralGalaxy_eqFunction_9210(data, threadData);
  SpiralGalaxy_eqFunction_9209(data, threadData);
  SpiralGalaxy_eqFunction_9208(data, threadData);
  SpiralGalaxy_eqFunction_1931(data, threadData);
  SpiralGalaxy_eqFunction_9203(data, threadData);
  SpiralGalaxy_eqFunction_1933(data, threadData);
  SpiralGalaxy_eqFunction_9202(data, threadData);
  SpiralGalaxy_eqFunction_1935(data, threadData);
  SpiralGalaxy_eqFunction_9201(data, threadData);
  SpiralGalaxy_eqFunction_1937(data, threadData);
  SpiralGalaxy_eqFunction_9214(data, threadData);
  SpiralGalaxy_eqFunction_9215(data, threadData);
  SpiralGalaxy_eqFunction_1940(data, threadData);
  SpiralGalaxy_eqFunction_1941(data, threadData);
  SpiralGalaxy_eqFunction_9216(data, threadData);
  SpiralGalaxy_eqFunction_9217(data, threadData);
  SpiralGalaxy_eqFunction_9220(data, threadData);
  SpiralGalaxy_eqFunction_9219(data, threadData);
  SpiralGalaxy_eqFunction_9218(data, threadData);
  SpiralGalaxy_eqFunction_1947(data, threadData);
  SpiralGalaxy_eqFunction_9213(data, threadData);
  SpiralGalaxy_eqFunction_1949(data, threadData);
  SpiralGalaxy_eqFunction_9212(data, threadData);
  SpiralGalaxy_eqFunction_1951(data, threadData);
  SpiralGalaxy_eqFunction_9211(data, threadData);
  SpiralGalaxy_eqFunction_1953(data, threadData);
  SpiralGalaxy_eqFunction_9224(data, threadData);
  SpiralGalaxy_eqFunction_9225(data, threadData);
  SpiralGalaxy_eqFunction_1956(data, threadData);
  SpiralGalaxy_eqFunction_1957(data, threadData);
  SpiralGalaxy_eqFunction_9226(data, threadData);
  SpiralGalaxy_eqFunction_9227(data, threadData);
  SpiralGalaxy_eqFunction_9230(data, threadData);
  SpiralGalaxy_eqFunction_9229(data, threadData);
  SpiralGalaxy_eqFunction_9228(data, threadData);
  SpiralGalaxy_eqFunction_1963(data, threadData);
  SpiralGalaxy_eqFunction_9223(data, threadData);
  SpiralGalaxy_eqFunction_1965(data, threadData);
  SpiralGalaxy_eqFunction_9222(data, threadData);
  SpiralGalaxy_eqFunction_1967(data, threadData);
  SpiralGalaxy_eqFunction_9221(data, threadData);
  SpiralGalaxy_eqFunction_1969(data, threadData);
  SpiralGalaxy_eqFunction_9234(data, threadData);
  SpiralGalaxy_eqFunction_9235(data, threadData);
  SpiralGalaxy_eqFunction_1972(data, threadData);
  SpiralGalaxy_eqFunction_1973(data, threadData);
  SpiralGalaxy_eqFunction_9236(data, threadData);
  SpiralGalaxy_eqFunction_9237(data, threadData);
  SpiralGalaxy_eqFunction_9240(data, threadData);
  SpiralGalaxy_eqFunction_9239(data, threadData);
  SpiralGalaxy_eqFunction_9238(data, threadData);
  SpiralGalaxy_eqFunction_1979(data, threadData);
  SpiralGalaxy_eqFunction_9233(data, threadData);
  SpiralGalaxy_eqFunction_1981(data, threadData);
  SpiralGalaxy_eqFunction_9232(data, threadData);
  SpiralGalaxy_eqFunction_1983(data, threadData);
  SpiralGalaxy_eqFunction_9231(data, threadData);
  SpiralGalaxy_eqFunction_1985(data, threadData);
  SpiralGalaxy_eqFunction_9244(data, threadData);
  SpiralGalaxy_eqFunction_9245(data, threadData);
  SpiralGalaxy_eqFunction_1988(data, threadData);
  SpiralGalaxy_eqFunction_1989(data, threadData);
  SpiralGalaxy_eqFunction_9246(data, threadData);
  SpiralGalaxy_eqFunction_9247(data, threadData);
  SpiralGalaxy_eqFunction_9250(data, threadData);
  SpiralGalaxy_eqFunction_9249(data, threadData);
  SpiralGalaxy_eqFunction_9248(data, threadData);
  SpiralGalaxy_eqFunction_1995(data, threadData);
  SpiralGalaxy_eqFunction_9243(data, threadData);
  SpiralGalaxy_eqFunction_1997(data, threadData);
  SpiralGalaxy_eqFunction_9242(data, threadData);
  SpiralGalaxy_eqFunction_1999(data, threadData);
  SpiralGalaxy_eqFunction_9241(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif