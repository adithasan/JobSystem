// JobSystem.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#pragma optimize( "", off )
#define CDS_JOB_IMPLEMENTATION
#include "cds_job.h"
#include <thread>
#include <chrono>
#include "IEJob.h"

static void WorkerThreadFunction(JobSystem::JobSystemContext* jobSysContext, unsigned int workerId)
{
	assert(JobSystem::InitializeWorker(jobSysContext, workerId));
	JobSystem::WaitUntilJobSystemTermination(jobSysContext);
}

static void WorkTask(JobSystem::Job *job, const void*data)
{
	(void)job;
	(void)data;
	int result;
	for (int j = 0; j < 10000; j++)
	{
		for (int i = 0; i < 10000; i++)
		{
			result = i * 2;
		}
	}
}

static void RootJobFunction(JobSystem::Job *job, const void*data)
{
	for (int k = 0; k < 24; k++)
	{
		JobSystem::Job* workerJob = JobSystem::CreateJobAsChild(job, WorkTask);
		JobSystem::KickJob(workerJob);
	}
}

int main()
{
	const int numCores = 12;
	std::cout<< "Number of system cores %d\n", numCores;
	std::cout << "Serial workload started\n";
    auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 24; i++)
	{
		WorkTask(nullptr, nullptr);
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "Serial workload completed in "<< (double)elapsedMillis << " ms\n";

	std::cout << "Parallel workload started\n";
	JobSystem::JobSystemContext* jobContext = new JobSystem::JobSystemContext(numCores);

	JobSystem::InitializeWorker(jobContext, 0);
	std::thread workerThreads[numCores - 1];
	// Set up threads
	for (int iThread = 0; iThread < numCores - 1; iThread++)
	{
		workerThreads[iThread] = std::thread(WorkerThreadFunction, jobContext, iThread + 1);
	}
	startTime = std::chrono::high_resolution_clock::now();
	auto masterJob = JobSystem::CreateJob(RootJobFunction);
	JobSystem::KickJob(masterJob);
	JobSystem::WaitForJob(masterJob);
	jobContext->Deactivate();
	for (int iThread = 0; iThread < numCores - 1; iThread++)
	{
		workerThreads[iThread].join();
	}
	endTime = std::chrono::high_resolution_clock::now();
	elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "Parallel workload completed in " << (double)elapsedMillis << " ms\n";

	std::cin.get();
}

