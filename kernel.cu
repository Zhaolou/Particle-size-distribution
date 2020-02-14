// GPUMieScattering.cpp : 定义控制台应用程序的入口点。
//
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <device_functions.h>
#include <time.h>
#include <math.h>
#define PI 3.141592653589793

__global__ void kernel(int wn, double* wavelength_data, double* tI, int populationSize, double* op0, double* op1, double* op2, double* op3, double* op4, double* op5, 
	float radiusMax, float radiusMin, double* fitnessValue, float delta_Radius,
	double *spectralExtinctionAll, double *spectralExtinctionDataBase, 
	int dataBaseTerm, int psdType)
{

    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	int i, j;
	float radius;
	float spectralExtinctionSingle;
	float psd;
	float scale0, scale1;
	int index;
	if(tid < populationSize)
	{
		fitnessValue[tid] =0;
		for(j = 0; j < wn; j++)
		{
			spectralExtinctionAll[tid*wn + j] = 0;
		}
		index = 0;
		for(radius = radiusMin; radius < radiusMax; radius = radius + delta_Radius)
		{				
			if(psdType == 0)		//R-R
			{			//op0: N0, op1: R, op2: sigma, op3: n0, op4: R2, op5: sigma2
				psd = 1 * op2[tid] / op1[tid] * pow (radius / op1[tid], op2[tid] - 1) * exp(-pow(radius/op1[tid], op2[tid]));
			}
			else if(psdType == 1)   //N-N
			{
				psd = 1 / sqrt(2*PI) / op2[tid] * exp(-(radius-op1[tid]) * (radius-op1[tid])/2/(op2[tid] * op2[tid]));
			}
			else //bimodal
			{
				psd =1 * (op3[tid] * op2[tid] / op1[tid] * pow(radius/op1[tid], op2[tid] - 1) * exp(-pow(radius/op1[tid], op2[tid]))
					+ (1-op3[tid]) * op5[tid] / op4[tid] * pow(radius/op4[tid], op5[tid] - 1) * exp(-pow(radius/op4[tid], op5[tid])));	
			}
			for(j = 0; j < wn; j++)		
			{
				spectralExtinctionAll[j + tid * wn] = spectralExtinctionAll[j + tid * wn] + spectralExtinctionDataBase[index + j * dataBaseTerm] * psd / radius * delta_Radius;	
			}	
			index++;
		}
		scale0 = 0; scale1 = 0;
		for(j = 0; j < wn; j++)
		{
			scale0 = scale0 + tI[j];
			scale1 = scale1 + spectralExtinctionAll[j + tid * wn];
		}		
		for(j = 0; j < wn; j++)
		{
			fitnessValue[tid] =  fitnessValue[tid] + abs(spectralExtinctionAll[j + tid * wn] * scale0/scale1 - tI[j]);
		}
	}	
}

//memory on device
double* wavelength_data;
double* mr_data;
double* mi_data;
double* tI_data;
double* op0_data;
double* op1_data;
double* op2_data;
double* op3_data;
double* op4_data;
double* op5_data;
double* abcdr_data;
double* abcdi_data;
double* fitnessValue_data;

double *bxr_data, *bxi_data, *bzr_data, *bzi_data, *yxr_data, *yxi_data, *hxr_data, *hxi_data, *axr_data, *axi_data, *azr_data, *azi_data, *ahxr_data, *ahxi_data;
double* spectralExtinctionAll_data;
double* spectralExtinctionDataBase_data;



//memory on the host machine
double wavelength[20], mr[20], mi[20], tI[20], op0[20000], op0range[2], op1range[2], op2range[2], op3range[2], op4range[2], op5range[2], fitnessValue[20000];
double bxr[2000], bxi[2000], bzr[2000], bzi[2000], yxr[2000], yxi[2000], hxr[2000], hxi[2000], axr[2000], axi[2000], azr[2000], azi[2000], ahxr[2000], ahxi[2000], abcdr[4000], abcdi[4000];
double spectralExtinctionAll[100000];
double spectralExtinctionDataBase[80000];
int dataBaseTerm;
double op1[20000];
double op2[20000];
double op3[20000];
double op4[20000];
double op5[20000];

cudaEvent_t start, stop;  
float processingTime;


int populationSize;
double globalBestFitnessValue;
double gbParticlePosition[5];
double localBestFitnessValue[20000];
double lbop1[20000];
double lbop2[20000];
double lbop3[20000];
double lbop4[20000];
double lbop5[20000];
double vop1[20000];
double vop2[20000];
double vop3[20000];
double vop4[20000];
double vop5[20000];



int GPUExit()
{
	cudaEventDestroy(start);  
	cudaEventDestroy(stop);


    checkCudaErrors(cudaFree(mr_data));
    checkCudaErrors(cudaFree(mi_data));
    checkCudaErrors(cudaFree(wavelength_data));
    checkCudaErrors(cudaFree(tI_data));
    checkCudaErrors(cudaFree(op0_data));
    checkCudaErrors(cudaFree(op1_data));
    checkCudaErrors(cudaFree(op2_data));
    checkCudaErrors(cudaFree(op3_data));
    checkCudaErrors(cudaFree(op4_data));
    checkCudaErrors(cudaFree(op5_data));
    checkCudaErrors(cudaFree(fitnessValue_data));
    checkCudaErrors(cudaFree(abcdr_data));
	checkCudaErrors(cudaFree(abcdi_data));



    checkCudaErrors(cudaFree(bxr_data));
    checkCudaErrors(cudaFree(bxi_data));
    checkCudaErrors(cudaFree(bzr_data));
    checkCudaErrors(cudaFree(bzi_data));
    checkCudaErrors(cudaFree(yxr_data));
    checkCudaErrors(cudaFree(yxi_data));
    checkCudaErrors(cudaFree(hxr_data));
    checkCudaErrors(cudaFree(hxi_data));
    checkCudaErrors(cudaFree(axr_data));
    checkCudaErrors(cudaFree(axi_data));
	checkCudaErrors(cudaFree(azr_data));

    checkCudaErrors(cudaFree(azi_data));
    checkCudaErrors(cudaFree(ahxr_data));
    checkCudaErrors(cudaFree(ahxi_data));
    checkCudaErrors(cudaFree(spectralExtinctionAll_data));
		


	return true;
}


int GPUInitialization( )
{
	int devID = -5;
	devID = findCudaDevice(0, NULL);
	
    checkCudaErrors(cudaMalloc((void **) &mr_data, 8*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &mi_data, 8*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &wavelength_data, 8*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &tI_data, 8*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &op0_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &op1_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &op2_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &op3_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &op4_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &op5_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &fitnessValue_data, 20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &abcdr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &abcdi_data, 200*20000*sizeof(double)));
    
    checkCudaErrors(cudaMalloc((void **) &bxr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &bxi_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &bzr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &bzi_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &yxr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &yxi_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &hxr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &hxi_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &axr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &axi_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &azr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &azi_data, 200*20000*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **) &ahxr_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &ahxi_data, 200*20000*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &spectralExtinctionAll_data, 200*20000*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **) &spectralExtinctionDataBase_data, 1000000*sizeof(double)));

	float time;  
	cudaEventCreate(&start);  
	cudaEventCreate(&stop); 
	return devID;
}
float GetProcessingTime()
{
	return processingTime;
}



int GPUMie_S12(int wn, double* wavelength, double* mr, double* mi, double* tI, int populationSize, double* op0, double* op1, double* op2, double* op3, double* op4, double* op5, 
	double radiusMax, double radiusMin, double* fitnessValue, double delta_Radius, int psdType)
{
	const unsigned int N = populationSize;
	cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy(wavelength_data, wavelength, wn*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(mr_data, mr, wn*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(mi_data, mi, wn*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(tI_data, tI, wn*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(op0_data, op0, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(op1_data, op1, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(op2_data, op2, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(op3_data, op3, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(op4_data, op4, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(op5_data, op5, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(fitnessValue_data, fitnessValue, populationSize*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(spectralExtinctionDataBase_data, spectralExtinctionDataBase, wn*dataBaseTerm*sizeof(double), cudaMemcpyHostToDevice));

	dim3 grid(N/192+1, 1, 1);
    dim3 block(12, 16, 1);

	kernel<<< grid, block >>>(wn, wavelength_data, tI_data, populationSize, op0_data, op1_data, op2_data, op3_data, op4_data, op5_data, 
		radiusMax, radiusMin, fitnessValue_data, delta_Radius,
		spectralExtinctionAll_data, spectralExtinctionDataBase_data,
		dataBaseTerm, psdType);



	checkCudaErrors(cudaMemcpy(fitnessValue, fitnessValue_data, populationSize*sizeof(double), cudaMemcpyDeviceToHost));
	cudaEventRecord(stop, 0);  
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&processingTime, start, stop);  
	return true;
}




void cpusqrt(double real, double image, double* r, double *i)
{
	double a = sqrt(sqrt(real*real + image*image));
	double angle = atan2(image, real)/2;
	(*r) = a * cos(angle);
	(*i) = a * sin(angle);
}

void cpuexp(double real, double image, double* r, double *i)
{
	(*r) = exp(real) * cos(image);
	(*i) = exp(real) * sin(image);
}


void cpusin(double real, double image, double* r, double *i)
{
	double a1r, a1i, a2r, a2i;
	cpuexp(-image, real, &a1r, &a1i);
	cpuexp(image, -real, &a2r, &a2i);
	(*r) = (a1i-a2i) / 2;
	(*i) = (a1r-a2r) / (-2);
}


void cpucos(double real, double image, double* r, double *i)
{
	double a1r, a1i, a2r, a2i;
	cpuexp(-image, real, &a1r, &a1i);
	cpuexp(image, -real, &a2r, &a2i);
	(*r) = (a1r+a2r) / 2;
	(*i) = (a1i+a2i) / 2;
}

void cpudivide(double r0, double i0, double r1, double i1, double* r, double *i)
{
	double rt, it;
	rt = r1/sqrt(r1*r1 + i1*i1);
	it = -i1/sqrt(r1*r1 + i1*i1);

	(*r) = r0*rt - i0*it;
	(*i) = r0*it + rt*i0;;
}

void cpumultiply(double r0, double i0, double rt, double it, double* r, double *i)
{
	(*r) = r0*rt - i0*it;
	(*i) = r0*it + rt*i0;;
}





int CPUMie_S12(int wn, int populationSize, double radiusMax, double radiusMin, double delta_Radius, int psdType = 0)
{
	double x;
	int i, j, tid;
	double radius;
	double r0, r1, r2, i0, i1, i2;
	double psd;
	double scale0, scale1;
	int index = 0;

	for(tid = 0; tid < populationSize; tid++)
	{
		fitnessValue[tid] =0;
		index = 0;
		for(j = 0; j < wn; j++)
		{
			spectralExtinctionAll[tid*wn + j] = 0;
		}
		for(radius = radiusMin; radius < radiusMax; radius = radius + delta_Radius)
		{		
			spectralExtinctionAll[j + tid * wn] = 0;
			if(psdType == 0)		//R-R
			{			//op0: N0, op1: R, op2: sigma, op3: n0, op4: R2, op5: sigma2
				psd = op2[tid] / op1[tid] * pow (radius / op1[tid], op2[tid] - 1) * exp(-pow(radius/op1[tid], op2[tid]));
			}
			else if(psdType == 1)   //N-N
			{
				psd = 1 / sqrt(2*PI) / op2[tid] * exp(-(radius-op1[tid]) * (radius-op1[tid])/2/(op2[tid] * op2[tid]));
			}
			else //bimodal
			{
				psd =1 * (op3[tid] * op2[tid] / op1[tid] * pow(radius/op1[tid], op2[tid] - 1) * exp(-pow(radius/op1[tid], op2[tid]))
					+ (1-op3[tid]) * op5[tid] / op4[tid] * pow(radius/op4[tid], op5[tid] - 1) * exp(-pow(radius/op4[tid], op5[tid])));	
			}	
			for(j = 0; j < wn; j++)
			{	
				spectralExtinctionAll[j + tid * wn] = spectralExtinctionAll[j + tid * wn] + spectralExtinctionDataBase[j*dataBaseTerm + index] * psd / radius * delta_Radius;
			}
			index++;
		}		
		scale0 = 0; scale1 = 0;
		for(j = 0; j < wn; j++)
		{
			scale0 = scale0 + tI[j];
			scale1 = scale1 + spectralExtinctionAll[j + tid * wn];
		}
		for(j = 0; j < wn; j++)
		{
			fitnessValue[tid] = fitnessValue[tid]  + abs(spectralExtinctionAll[j + tid * wn] * scale0/scale1 - tI[j]);
		}

	}	
	return true;
}

void CPUMieGeneration(int wn, double* wavelength, double* mr, double* mi, double radiusMax, double radiusMin, double delta_Radius) 
{
	double wavelength0;
	int nmax;
	double x;
	double mr0, mi0;
	double u;
	double sqx;
	double sqzr, sqzi;
	double bxp05, bxm05, yxp05, yxm05, bzp05r, bzp05i, bzm05r, bzm05i;
	double r1temp, i1temp, r2temp, i2temp, r3temp, i3temp;
	double dt;
	double ctr, cti;
	double zr, zi, m2r, m2i,recip_zr, recip_zi;
	int i, j, index = 0;
	double radius;
	double r0, r1, r2, i0, i1, i2;
	double spectralExtinctionSingle;
	double rhor, rhoi;
	double psd;
	double scale0, scale1;
	dataBaseTerm = int((radiusMax - radiusMin) / delta_Radius) + 1;
	double rrr[10000];
	FILE* fp = fopen("d:\\gg.txt", "w+");

	for(j = 0; j < wn; j++)
	{
		mr0 = mr[j]; mi0 = mi[j];
		wavelength0 = wavelength[j];
		index = 0;
		for(radius = radiusMin; radius < radiusMax; radius = radius + delta_Radius)
		{
			spectralExtinctionDataBase[j * dataBaseTerm + index]= 0;

			spectralExtinctionSingle = 0;
			x = 2*PI/wavelength0 * radius;			
			zr = mr0*x; zi = mi0*x;
			recip_zr = zr / (zr*zr + zi*zi);
			recip_zi = -zi / (zr*zr + zi*zi);
			m2r = mr0*mr0-mi0*mi0; m2i = 2*mr0*mi0;
			sqx = sqrt(0.5*PI/x);
			cpusqrt(recip_zr, recip_zi, &sqzr, &sqzi);
			sqzr = sqzr * sqrt(0.5*PI);
			sqzi = sqzi * sqrt(0.5*PI);
			bxp05 = sin(x) * sqrt(2/PI/x);
			bxm05 = cos(x) * sqrt(2/PI/x);
			yxp05 = bxp05;
			yxm05 = bxm05;
			cpusin(zr, zi, &r1temp, &i1temp);
			cpusqrt(recip_zr, recip_zi, &r2temp, &i2temp);
			bzp05r = sqrt(2/PI) * (r1temp*r2temp - i1temp*i2temp);
			bzp05i = sqrt(2/PI) * (r1temp*i2temp + r2temp*i1temp);
			cpucos(zr, zi, &r1temp, &i1temp);
			bzm05r = sqrt(2/PI) * (r1temp*r2temp - i1temp*i2temp);
			bzm05i = sqrt(2/PI) * (r1temp*i2temp + r2temp*i1temp);

			nmax = (int)(2+x+4*pow(x, 0.333333));
			for(i = 0; i < nmax; i++)
			{
				dt = 2.0 / x * (i + 0.5) * bxp05 - bxm05;
				bxm05 = bxp05;
				bxp05 = dt;
				bxr[i] = bxp05 * sqx;
				bxi[i] = 0;
				ctr = (2*i+1) *(bzp05r*recip_zr - bzp05i*recip_zi) - bzm05r;
				cti = (2*i+1) * (bzp05r*recip_zi+bzp05i*recip_zr) - bzm05i;
				bzm05r= bzp05r;
				bzm05i = bzp05i;
				bzp05r = ctr;
				bzp05i = cti;
				bzr[i] = bzp05r*sqzr - bzp05i*sqzi;
				bzi[i] = bzp05r*sqzi+bzp05i*sqzr;
				dt = 2.0/x*(-i-0.5)*yxm05-yxp05;
				yxp05 = yxm05;
				yxm05= dt;
				yxr[i] = (bxp05*cos((i+1.5)*PI) - yxm05) / sin((i+1.5)*PI) * sqx;
				yxi[i] = 0;
				hxr[i] = bxr[i];
				hxi[i] = yxr[i];
			}
			axr[0] = sin(x) - bxr[0]; axi[0] = 0;
			cpusin(zr,zi,&(azr[0]), &(azi[0]));
			azr[0] = azr[0] - bzr[0];
			azi[0] = azi[0] - bzi[0];
			ahxr[0] = sin(x) - hxr[0];
			ahxi[0] = -cos(x) - hxi[0];
			for(i = 1; i < nmax; i++)
			{
				axr[i] = x*bxr[i-1] - (i+1) *bxr[i]; 
				axi[i] = 0;
				azr[i] = zr*bzr[i-1] -zi*bzi[i-1] - (i+1)*bzr[i];
				azi[i] = zr*bzi[i-1] + zi*bzr[i-1] -(i+1)*bzi[i];
				ahxr[i] = x*hxr[i-1] - (i+1)*hxr[i];
				ahxi[i] = x*hxi[i-1] - (i+1)*hxi[i];
			}
			for(i = 1; i < nmax; i++)
			{
				cpumultiply(m2r, m2i, bzr[i], bzi[i], &r0, &i0);
				cpumultiply(r0,  i0,  ahxr[i],ahxi[i],&r1, &i1);
				cpumultiply(hxr[i],hxi[i], azr[i], azi[i], &r0, &i0);
				r2 = r1 - r0; i2 = i1 - i0;			
				r0 = (m2r*bzr[i] - m2i*bzi[i]) * axr[i] - bxr[i] * azr[i];
				i0 = (m2r*bzi[i] + m2i*bzr[i]) * axr[i] - bxr[i] * azi[i];
				cpudivide(r0, i0, r2, i2, &(abcdr[0*nmax + i]), &(abcdi[0*nmax + i]));

				cpumultiply(bzr[i], bzi[i], ahxr[i],ahxi[i], &r1, &i1);
				cpumultiply(hxr[i],hxi[i], azr[i], azi[i], &r0, &i0);
				r2 = r1 - r0; i2 = i1 - i0;			
				r0 = bzr[i] * axr[i] - bxr[i] * azr[i];
				i0 = bzi[i] * axr[i] - bxr[i] * azi[i];
				cpudivide(r0, i0, r2, i2, &(abcdr[1*nmax + i]), &(abcdi[1*nmax + i]));
				spectralExtinctionSingle = spectralExtinctionSingle + 2 / x / x * (2 * i + 1) * (abcdr[0*nmax + i] + abcdr[1*nmax + i]);
			}
			spectralExtinctionDataBase[j * dataBaseTerm + index]= spectralExtinctionSingle;
			rrr[j * dataBaseTerm + index]= radius*1e6;
			index++;
		}
	}		
	for(i = 0; i < wn*dataBaseTerm;i++)
	{
		fprintf(fp,"%f %f\n",rrr[i], spectralExtinctionDataBase[i]);
	}
	fclose(fp);
	return;
}


void UpdateFitnessValue()
{
    for (int i = 0; i < populationSize; i++)
    {
        if (globalBestFitnessValue > fitnessValue[i])
        {
            globalBestFitnessValue = fitnessValue[i];
            gbParticlePosition[0] = op1[i];
            gbParticlePosition[1] = op2[i];
            gbParticlePosition[2] = op3[i];
            gbParticlePosition[3] = op4[i];
            gbParticlePosition[4] = op5[i];
        }
        if (localBestFitnessValue[i] > fitnessValue[i])
        {
            localBestFitnessValue[i] = fitnessValue[i];
            lbop1[i] = op1[i];
            lbop2[i] = op2[i];
            lbop3[i] = op3[i];
            lbop4[i] = op4[i];
            lbop5[i] = op5[i];
        }
    }
}
double cpumax(double x, double y)
{
	return x > y ? x : y;
}
double cpumin(double x, double y)
{
	return x < y ? x : y;
}


void UpdateParticlePosition()
{
    double c1, c2, c3, c4, c5, c6, c7, c8;
    double velocity = 0;         
	srand(time(0));

    for (int i = 0; i < populationSize; i++)
    {
        c1 = 1e-1 *  rand() / (RAND_MAX + 1.0);
        c2 = 1e-1 *  rand() / (RAND_MAX + 1.0);
        c3 = 1e-1 *  rand() / (RAND_MAX + 1.0);
		
		c4 = 5e-2 *  (rand() / (RAND_MAX + 1.0) - 0.5);
		c5 = 5e-2 *  (rand() / (RAND_MAX + 1.0) - 0.5);
		c6 = 5e-2 *  (rand() / (RAND_MAX + 1.0) - 0.5);
		c7 = 5e-2 *  (rand() / (RAND_MAX + 1.0) - 0.5);
		c8 = 5e-2 *  (rand() / (RAND_MAX + 1.0) - 0.5);
		vop1[i] = c1*vop1[i] + c2 * (gbParticlePosition[0] - op1[i]) + c3 * (lbop1[i] - op1[i]) + c4*1e-6;
		vop2[i] = c1*vop2[i] + c2 * (gbParticlePosition[1] - op2[i]) + c3 * (lbop2[i] - op2[i]) + c5;
		vop3[i] = c1*vop3[i] + c2 * (gbParticlePosition[2] - op3[i]) + c3 * (lbop3[i] - op3[i]) + c6;
		vop4[i] = c1*vop4[i] + c2 * (gbParticlePosition[3] - op4[i]) + c3 * (lbop4[i] - op4[i]) + c7*1e-6;
		vop5[i] = c1*vop5[i] + c2 * (gbParticlePosition[4] - op5[i]) + c3 * (lbop5[i] - op5[i]) + c8;
		
		velocity = sqrt(vop1[i]*vop1[i]*1e12 + vop2[i]*vop2[i] + vop3[i]*vop3[i] + vop4[i]*vop4[i]*1e12 + vop5[i]*vop5[i]);
        if(velocity > 0.2)
        {
            vop1[i] = vop1[i] * 0.2 / velocity;
            vop2[i] = vop2[i] * 0.2 / velocity;
            vop3[i] = vop3[i] * 0.2 / velocity;
            vop4[i] = vop4[i] * 0.2 / velocity;
            vop5[i] = vop5[i] * 0.2 / velocity;           
		}
        op1[i] = cpumin(cpumax(op1[i] + vop1[i], op1range[0]), op1range[1]);
        op2[i] = cpumin(cpumax(op2[i] + vop2[i], op2range[0]), op2range[1]);
        op3[i] = cpumin(cpumax(op3[i] + vop3[i], op3range[0]), op3range[1]);
        op4[i] = cpumin(cpumax(op4[i] + vop4[i], op4range[0]), op4range[1]);
        op5[i] = cpumin(cpumax(op5[i] + vop5[i], op5range[0]), op5range[1]);                
    }
}

int main()
{
	int ProcessingType = 0;
	const int CPU = 0;
	const int GPU = 1;
	int wn = 4;
	int maxGen = 30; 
	double nrand;
	double radiusMax = 15e-6;
	double radiusMin = 0.1e-6;
	double delta_Radius = 0.2e-7;
	int generationNumber = 50;
	int psdType = 0;


	
	populationSize = 2000;
	globalBestFitnessValue = 1000;

	for(int i = 0; i < populationSize; i++)
	{
		localBestFitnessValue[i] = 1000;
		vop1[i] = 0;
		vop2[i] = 0;
		vop3[i] = 0;
		vop4[i] = 0;
		vop5[i] = 0;
	}
	printf("Please input the wavelength number:");
	scanf(" %d", &wn);
	printf("Please input the population size:");
	scanf(" %d", &populationSize);
	int psdFunctionType = 0;
	wavelength[0] = 0.34e-6; wavelength[1] = 0.675e-6;  wavelength[2] = 0.87e-6; wavelength[3] = 1.64e-6;
	wavelength[4] = 0.34e-6; wavelength[5] = 0.675e-6;  wavelength[6] = 0.87e-6; wavelength[7] = 1.64e-6;
	wavelength[8] = 0.34e-6; wavelength[9] = 0.675e-6;  wavelength[10] = 0.87e-6; wavelength[11] = 1.64e-6;
	mr[0] = 1.5; mr[1] = 1.5; mr[2] = 1.5; mr[3] = 1.5;
	mr[4] = 1.5; mr[5] = 1.5; mr[6] = 1.5; mr[7] = 1.5;
	mr[8] = 1.5; mr[9] = 1.5; mr[10] = 1.5; mr[11] = 1.5;
	mi[0] = 0.008; mi[1] = 0.008; mi[2] = 0.008; mi[3] = 0.008;
	mi[4] = 0.008; mi[5] = 0.008; mi[6] = 0.008; mi[7] = 0.008;
	mi[8] = 0.008; mi[9] = 0.008; mi[10] = 0.008; mi[11] = 0.008;

	
	srand((unsigned int)time(0));
	
	tI[0] = 1.9162163514667161; tI[1] = 2.2859890084541550; tI[2] = 2.3775408000437976; tI[3] = 2.3032538400353300;
	tI[0] = 1.938; tI[1] = 2.257; tI[2] = 2.316; tI[3] = 2.372;
	tI[4] = 1.938; tI[5] = 2.257; tI[6] = 2.316; tI[7] = 2.372;
	tI[8] = 1.938; tI[9] = 2.257; tI[10] = 2.316; tI[11] = 2.372;

	GPUInitialization();
	op0range[0] = 0;		op0range[1] = 0;
	op1range[0] = 0.5e-6;		op1range[1] = 10e-6;
	op2range[0] = 1.1;		op2range[1] = 5;
	op3range[0] = 0;		op3range[1] = 1;
	op4range[0] = 0.5e-6;		op4range[1] = 10e-6;
	op5range[0] = 1.1;		op5range[1] = 5;
	for(int i = 0; i < populationSize; i++)
	{
		nrand = rand();
		op1[i] = op1range[0] + (op1range[1] - op1range[0]) * nrand / (RAND_MAX + 1.0);
		nrand = rand();
		op2[i] = op2range[0] + (op2range[1] - op2range[0]) * nrand / (RAND_MAX + 1.0);
		nrand = rand();
		op3[i] = op3range[0] + (op3range[1] - op3range[0]) * nrand / (RAND_MAX + 1.0);
		nrand = rand();
		op4[i] = op4range[0] + (op4range[1] - op4range[0]) * nrand / (RAND_MAX + 1.0);
		nrand = rand();
		op5[i] = op5range[0] + (op5range[1] - op5range[0]) * nrand / (RAND_MAX + 1.0);
	}
	CPUMieGeneration(wn, wavelength, mr, mi, radiusMax, radiusMin, delta_Radius); 
	FILE *fp = fopen("d:\\ff.txt", "w+");
	printf("Processing unit(0-CPU, 1-GPU:)");
	scanf("%d", &ProcessingType);

	printf("Particle size distribution function(0-RR, 1-NN, 2-Bimodal:)");
	scanf(" %d", &psdType);

		DWORD tic = GetTickCount();
	for (int gn = 0; gn < generationNumber; gn++)
    {
        if (ProcessingType == CPU)                     
        {
            CPUMie_S12(wn, populationSize, radiusMax, radiusMin, delta_Radius, psdType);   
		}
        else
        { 
			GPUMie_S12(wn, wavelength, mr, mi, tI, populationSize, op0, op1, op2, op3, op4, op5, 
				radiusMax, radiusMin, fitnessValue, delta_Radius, psdType);
        }
        UpdateFitnessValue();
        UpdateParticlePosition();
		double averageFitnessValue = 0;
		for(int i = 0; i < populationSize; i++)
			averageFitnessValue = averageFitnessValue + fitnessValue[i];
		averageFitnessValue = averageFitnessValue/populationSize;
		fprintf(fp,"%f %f\n", globalBestFitnessValue, averageFitnessValue);
    }
	fclose(fp);
	DWORD toc = GetTickCount()-tic;
	printf("Elapsed time is %d ms\n", toc);
	op1[0] = gbParticlePosition[0];
	op2[0] = gbParticlePosition[1];
	op3[0] = gbParticlePosition[2];
	op4[0] = gbParticlePosition[3];
	op5[0] = gbParticlePosition[4];



	CPUMie_S12(wn, 1, radiusMax, radiusMin, delta_Radius, psdType); 
	double retrievedVector[20];
	double targetVector[20];
	double sum0 = 0, sum1 = 0;
	for(int i = 0; i < wn; i++)
	{
		retrievedVector[i] = spectralExtinctionAll[i];
		targetVector[i] = tI[i];
		sum0 = sum0 + retrievedVector[i];
		sum1 = sum1 + targetVector[i];
	}
	for(int i = 0; i < wn; i++)
	{
		retrievedVector[i] = spectralExtinctionAll[i] * sum1 / sum0;
	}
	GPUExit();
    return 0;
}
