/* Jacobians 7 */
#include "PMSM_Plant_FMU_model.h"
#include "PMSM_Plant_FMU_12jac.h"
#include "simulation/jacobian_util.h"
#include "util/omc_file.h"
int PMSM_Plant_FMU_functionJacADJ_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  return 0;
}

void PMSM_Plant_FMU_JacADJ_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian) { /* empty */ }

int PMSM_Plant_FMU_functionJacH_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  return 0;
}

void PMSM_Plant_FMU_JacH_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian) { /* empty */ }

int PMSM_Plant_FMU_functionJacF_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  return 0;
}

void PMSM_Plant_FMU_JacF_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian) { /* empty */ }

int PMSM_Plant_FMU_functionJacD_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  return 0;
}

void PMSM_Plant_FMU_JacD_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian) { /* empty */ }

int PMSM_Plant_FMU_functionJacC_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  return 0;
}

void PMSM_Plant_FMU_JacC_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian) { /* empty */ }

int PMSM_Plant_FMU_functionJacB_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  return 0;
}

void PMSM_Plant_FMU_JacB_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian) { /* empty */ }

/* constant equations */
/* dynamic equations */

OMC_DISABLE_OPT
int PMSM_Plant_FMU_functionJacA_constantEqns(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  int index = PMSM_Plant_FMU_INDEX_JAC_A;
  
  
  return 0;
}

int PMSM_Plant_FMU_functionJacA_column(DATA* data, threadData_t *threadData, JACOBIAN *jacobian, JACOBIAN *parentJacobian)
{
  int index = PMSM_Plant_FMU_INDEX_JAC_A;
  
  
  return 0;
}

void PMSM_Plant_FMU_JacA_DAG(DATA* data, threadData_t* threadData, JACOBIAN* jacobian)
{
  const size_t eqMap[] = {};
  buildEvalDAG_Jac(jacobian, data->modelData, sizeof(eqMap)/sizeof(size_t), eqMap);
}

int PMSM_Plant_FMU_initialAnalyticJacobianADJ(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  jacobian->availability = JACOBIAN_NOT_AVAILABLE;
  return 1;
}
int PMSM_Plant_FMU_initialAnalyticJacobianH(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  jacobian->availability = JACOBIAN_NOT_AVAILABLE;
  return 1;
}
int PMSM_Plant_FMU_initialAnalyticJacobianF(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  jacobian->availability = JACOBIAN_NOT_AVAILABLE;
  return 1;
}
int PMSM_Plant_FMU_initialAnalyticJacobianD(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  jacobian->availability = JACOBIAN_NOT_AVAILABLE;
  return 1;
}
int PMSM_Plant_FMU_initialAnalyticJacobianC(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  jacobian->availability = JACOBIAN_NOT_AVAILABLE;
  return 1;
}
int PMSM_Plant_FMU_initialAnalyticJacobianB(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  jacobian->availability = JACOBIAN_NOT_AVAILABLE;
  return 1;
}
OMC_DISABLE_OPT
int PMSM_Plant_FMU_initialAnalyticJacobianA(DATA* data, threadData_t *threadData, JACOBIAN *jacobian)
{
  size_t count;

  FILE* pFile = openSparsePatternFile(data, threadData, "PMSM_Plant_FMU_JacA.bin");
  
  initJacobian(jacobian, 4, 4, 0, NULL, PMSM_Plant_FMU_functionJacA_column, NULL, NULL);
  jacobian->sparsePattern = allocSparsePattern(4, 12, 4);
  jacobian->availability = JACOBIAN_ONLY_SPARSITY;
  jacobian->isRowEval = 0 /* false */;
  
  /* read lead index of compressed sparse column */
  count = omc_fread(jacobian->sparsePattern->leadindex, sizeof(unsigned int), 4+1, pFile, FALSE);
  if (count != 4+1) {
    throwStreamPrint(threadData, "Error while reading lead index list of sparsity pattern. Expected %d, got %zu", 4+1, count);
  }
  
  /* read sparse index */
  count = omc_fread(jacobian->sparsePattern->index, sizeof(unsigned int), 12, pFile, FALSE);
  if (count != 12) {
    throwStreamPrint(threadData, "Error while reading row index list of sparsity pattern. Expected %d, got %zu", 12, count);
  }
  
  /* write color array */
  /* color 1 with 1 columns */
  readSparsePatternColor(threadData, pFile, jacobian->sparsePattern->colorCols, 1, 1, 4);
  /* color 2 with 1 columns */
  readSparsePatternColor(threadData, pFile, jacobian->sparsePattern->colorCols, 2, 1, 4);
  /* color 3 with 1 columns */
  readSparsePatternColor(threadData, pFile, jacobian->sparsePattern->colorCols, 3, 1, 4);
  /* color 4 with 1 columns */
  readSparsePatternColor(threadData, pFile, jacobian->sparsePattern->colorCols, 4, 1, 4);
  
  omc_fclose(pFile);
  
  return 0;
}


