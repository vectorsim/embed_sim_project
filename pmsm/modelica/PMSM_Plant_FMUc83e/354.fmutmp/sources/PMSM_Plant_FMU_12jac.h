/* Jacobians */
static _index_t one_dim[1] = { 1 };
static modelica_real nominal_data[1] = { 1.0 };
static modelica_real start_data[1]   = { 0.0 };
static modelica_real min_data[1]   = { -DBL_MAX };
static modelica_real max_data[1]   = { DBL_MAX };
static const REAL_ATTRIBUTE dummyREAL_ATTRIBUTE = {
  .unit = NULL,
  .displayUnit = NULL,
  .min = {
    .ndims     = 1,
    .dim_size  = one_dim,
    .data      = (void*) min_data,
    .flexible  = FALSE
  },
  .max = {
    .ndims     = 1,
    .dim_size  = one_dim,
    .data      = (void*) max_data,
    .flexible  = FALSE
  },
  .fixed = FALSE,
  .useNominal = FALSE,
  .nominal = {
    .ndims     = 1,
    .dim_size  = one_dim,
    .data      = (void*) nominal_data,
    .flexible  = FALSE
  },
  .start = {
    .ndims     = 1,
    .dim_size  = one_dim,
    .data      = (void*) start_data,
    .flexible  = FALSE
  }
};

#if defined(__cplusplus)
extern "C" {
#endif

/* Jacobian Variables */
#define PMSM_Plant_FMU_INDEX_JAC_ADJ 0
int PMSM_Plant_FMU_functionJacADJ_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianADJ(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacADJ_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);


#define PMSM_Plant_FMU_INDEX_JAC_H 1
int PMSM_Plant_FMU_functionJacH_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianH(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacH_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);


#define PMSM_Plant_FMU_INDEX_JAC_F 2
int PMSM_Plant_FMU_functionJacF_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianF(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacF_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);


#define PMSM_Plant_FMU_INDEX_JAC_D 3
int PMSM_Plant_FMU_functionJacD_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianD(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacD_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);


#define PMSM_Plant_FMU_INDEX_JAC_C 4
int PMSM_Plant_FMU_functionJacC_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianC(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacC_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);


#define PMSM_Plant_FMU_INDEX_JAC_B 5
int PMSM_Plant_FMU_functionJacB_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianB(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacB_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);


#define PMSM_Plant_FMU_INDEX_JAC_A 6
int PMSM_Plant_FMU_functionJacA_column(DATA* data, threadData_t *threadData, JACOBIAN *thisJacobian, JACOBIAN *parentJacobian);
int PMSM_Plant_FMU_initialAnalyticJacobianA(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);
void PMSM_Plant_FMU_JacA_DAG(DATA* data, threadData_t *threadData, JACOBIAN *jacobian);

#if defined(__cplusplus)
}
#endif
