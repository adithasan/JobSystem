#pragma once
#include <atomic>
#include <queue>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace JobSystem
{
	// Begin Interface -------------------------------------
	class JobSystemContext;
	struct Job;
	Job* CreateJob(JobFunction function);
	Job* CreateJobAsChild(Job* parent, JobFunction function);
	bool InitializeWorker(JobSystemContext* jobContext, unsigned int workerId);
	// TODO: Maybe we just want to reference jobs by their ids
	bool KickJob(Job* job);
	void WaitForJob(Job* job);
	void WaitUntilJobSystemTermination(JobSystemContext* jobContext);
	// End Interface ---------------------------------------

#define JOB_CACHE_LINE_BYTES 64
#define JOB_PADDING_BYTES ( (JOB_CACHE_LINE_BYTES) - (sizeof(JobFunction) + sizeof(struct Job*) + sizeof(void*) + sizeof(std::atomic_int_fast32_t)) )

#ifdef _MSC_VER
#   define JOB_ATTR_ALIGN(alignment) __declspec(align(alignment))
#else
#   define JOB_ATTR_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif

#if   defined(_MSC_VER)
#   if _MSC_VER < 1900
#       define IE_JOB_THREAD_LOCAL __declspec(thread)
#   else
#       define IE_JOB_THREAD_LOCAL thread_local
#   endif
#elif defined(__GNUC__)
#   define IE_JOB_THREAD_LOCAL __thread
#elif defined(__clang__)
#   if defined(__APPLE__) || defined(__MACH__)
#       define IE_JOB_THREAD_LOCAL __thread
#   else
#       define IE_JOB_THREAD_LOCAL thread_local
#   endif
#endif

#ifdef _MSC_VER
#   include <windows.h>
#   define JOB_YIELD() YieldProcessor()
#   define JOB_COMPILER_BARRIER _ReadWriteBarrier()
#   define JOB_MEMORY_BARRIER std::atomic_thread_fence(std::memory_order_seq_cst);
#else
#   include <emmintrin.h>
#   define JOB_YIELD() _mm_pause()
#   define JOB_COMPILER_BARRIER asm volatile("" ::: "memory")
#   define JOB_MEMORY_BARRIER asm volatile("mfence" ::: "memory")
#endif

static IE_JOB_THREAD_LOCAL int TLS_WORKER_ID = -1;
static IE_JOB_THREAD_LOCAL int TLS_ALLOCATED_JOB_COUNT = 0;
static IE_JOB_THREAD_LOCAL JobSystemContext* TLS_JOB_SYS_CONTEXT = nullptr;
static IE_JOB_THREAD_LOCAL Job* TLS_JOB_POOL = nullptr;


class WorkStealingQueue
{
	// This is a lock-free queue / fifo
	// that supports other threads stealing items from the queue
public:
	static size_t BufferSize(int capacity) { return capacity * sizeof(Job*); }

	// bufferSize is the size in bytes of the buffer
	// capacity is the number of elements that can exist in the buffer
	bool Initialize(int capacity, void* buffer, size_t bufferSize);
	bool Push(Job* job);
	Job* Pop();
	Job* Steal();
private:
	Job** mEntries;
	std::atomic<uint64_t> mTop;
	uint64_t mBottom;
	int mCapacity;
	const size_t MASK = mCapacity - 1;
};

// TODO: replace all asserts with IE_ASSERT
bool WorkStealingQueue::Initialize(int capacity, void* buffer, size_t bufferSize)
{
	assert((capacity & MASK) != 0); // Capacity must be a power of 2!
	size_t minBufferSize = BufferSize(capacity);
	assert(bufferSize >= minBufferSize); // Inadequate buffer size!

	uint8_t* bufferNext = (uint8_t*)buffer;
	mEntries = (Job**)bufferNext;
	bufferNext += capacity * sizeof(Job*);
	assert(bufferNext - (uint8_t*)buffer == (intptr_t)minBufferSize);

	for (int iEntry = 0; iEntry < capacity; iEntry += 1)
	{
		mEntries[iEntry] = nullptr;
	}

	mTop.store(0);
	mBottom = 0;
	mCapacity = capacity;

	return true;
}

bool WorkStealingQueue::Push(Job* job)
{
	uint64_t jobIndex = mBottom;
	// The MASK ensures we wrap around the ring buffer
	mEntries[jobIndex & MASK] = job;

	// On platforms with a weak memory model, we would need a full memory fence
	JOB_COMPILER_BARRIER;

	mBottom = jobIndex + 1;

	return true;
}

Job* WorkStealingQueue::Steal()
{
	uint64_t top = mTop;

	// Ensure the top is always read before the bottom
	// On platforms with a weak memory model, we would need a full memory fence
	JOB_COMPILER_BARRIER;
	uint64_t bottom = mBottom;

	if (top < bottom)
	{
		// We have a non-empty queue
		Job* job = mEntries[top & MASK];
		// CAS is an implicit compiler barrier
		if (!std::atomic_compare_exchange_strong(&mTop, &top, top + 1)) 
		{
			// A Steal()/Pop() operation from another thread got this entry first
			return nullptr; 
		}
		mEntries[top & MASK] = nullptr;
		return job;
	}
	return nullptr; // queue empty
}

Job* WorkStealingQueue::Pop()
{
	uint64_t bottom = mBottom - 1;
	mBottom = bottom;
	// Ensure mBottom is published before reading top;
	JOB_MEMORY_BARRIER;
	uint64_t top = mTop;
	if (top <= bottom)
	{
		// non-empty queue
		Job* job = mEntries[bottom & MASK];
		if (top != bottom)
		{
			// Still more items left in the queue
			return job;
		}
		// This is the last item in the queue
		if (!std::atomic_compare_exchange_strong(&mTop, &top, top + 1))
		{
			// A Steal() operation from another thread got this entry first
			job = nullptr;
		}
		mBottom = top + 1;
		return job;
	}
	else
	{
		// queue empty
		mBottom = top;
		return nullptr;
	}
}

typedef void(*JobFunction)(Job*, const void*);
typedef JOB_ATTR_ALIGN(JOB_CACHE_LINE_BYTES) struct Job
{
	JobFunction function;
	Job* parent;
	std::atomic_int_fast32_t unfinishedJobs;
	char padding[JOB_PADDING_BYTES];
	// This is optional and may only be used when padding 
	// isn't sufficient to store the function data
	void* data;
} Job;

class JobSystemContext
{
public:
	// disable default, copy and move constructors/operators
	JobSystemContext() = delete;
	JobSystemContext(JobSystemContext const&) = delete;
	JobSystemContext(JobSystemContext&&) = delete;	
	JobSystemContext& operator=(JobSystemContext const&) = delete;
	JobSystemContext& operator=(JobSystemContext&&) = delete;

	JobSystemContext(unsigned int numWorkerThreads);
	~JobSystemContext();

	void Deactivate();

	static constexpr size_t MAX_JOBS_PER_THREAD = 4096; // this might be too high???
	WorkStealingQueue** mWorkerJobQueues;

	std::atomic<bool> mIsJobSystemActive = false;
	unsigned int mNumWorkers;
	unsigned int nextWorkerId;

	// These 2 can be initialized from a custom memory allocator
	void* mJobPoolBuffer;
	void* mQueueEntryBuffer;
};


JobSystemContext::JobSystemContext(unsigned int numWorkerThreads)
	: mNumWorkers(numWorkerThreads), nextWorkerId(0), mWorkerJobQueues(nullptr)
{
	const size_t queueBufferSize = WorkStealingQueue::BufferSize(MAX_JOBS_PER_THREAD);
	// Spawn the appropriate threads
	// Initialize the queues
	// Make the threads wait on 
	mWorkerJobQueues = new WorkStealingQueue*[numWorkerThreads];
	const size_t jobPoolBufferSize = numWorkerThreads * MAX_JOBS_PER_THREAD * sizeof(Job) + JOB_CACHE_LINE_BYTES - 1;
	mJobPoolBuffer = malloc(jobPoolBufferSize);

	mQueueEntryBuffer = malloc(queueBufferSize*numWorkerThreads);

	mIsJobSystemActive.store(true);
	for (int iWorker = 0; iWorker < numWorkerThreads; ++iWorker)
	{
		mWorkerJobQueues[iWorker] = new WorkStealingQueue();
		mWorkerJobQueues[iWorker]->Initialize(MAX_JOBS_PER_THREAD, (void*)(intptr_t(mQueueEntryBuffer) + iWorker * queueBufferSize), queueBufferSize);
	}
}

JobSystemContext::~JobSystemContext()
{
	for (int iWorker = 0; iWorker < mNumWorkers; iWorker++)
	{
		delete mWorkerJobQueues[iWorker];
	}
	// These can go into a large multipurpose arena allocator
	delete[] mWorkerJobQueues;
	free(mQueueEntryBuffer);
	free(mJobPoolBuffer);
}

void JobSystemContext::Deactivate()
{
	mIsJobSystemActive.store(false);
}


namespace internal
{
	static inline Job* AllocateJob()
	{
		auto index = TLS_ALLOCATED_JOB_COUNT++;
		return &TLS_JOB_POOL[index & (TLS_JOB_SYS_CONTEXT->MAX_JOBS_PER_THREAD - 1)];
	}

	static inline void Finish(Job* job)
	{
		const int32_t unfinishedJobs = --job->unfinishedJobs;
		assert(unfinishedJobs >= 0); // unfinished job cannot be negative
		if (unfinishedJobs == 0 && job->parent)
		{
			Finish(job->parent);
		}
	}

	static inline void ExecuteJob(Job* job)
	{
		(job->function)(job, job->data);
		Finish(job);
	}

	static Job* GetValidJob()
	{
		WorkStealingQueue* myQueue = TLS_JOB_SYS_CONTEXT->mWorkerJobQueues[TLS_WORKER_ID];
		auto job = myQueue->Pop();
		if (!job)
		{
			int victimIndex = 1 + (rand() % TLS_JOB_SYS_CONTEXT->mNumWorkers - 1);
			assert(victimIndex <= (TLS_JOB_SYS_CONTEXT->mNumWorkers - 1));
			WorkStealingQueue* victimQueue = TLS_JOB_SYS_CONTEXT->mWorkerJobQueues[victimIndex];
			job = victimQueue->Steal();
			if (!job)
			{
				JOB_YIELD();
				return nullptr;
			}
		}
		return job;
	}

}
bool InitializeWorker(JobSystemContext* jobContext, unsigned int workerId)
{
	TLS_ALLOCATED_JOB_COUNT = 0;
	TLS_JOB_SYS_CONTEXT = jobContext;
	TLS_WORKER_ID = workerId;
	assert(TLS_WORKER_ID < jobContext->mNumWorkers);
	void* jobPoolBufferAligned = (void*)((uintptr_t(jobContext->mJobPoolBuffer) + JOB_CACHE_LINE_BYTES - 1) & ~(JOB_CACHE_LINE_BYTES - 1));
	assert((uintptr_t(jobPoolBufferAligned) % JOB_CACHE_LINE_BYTES) == 0);
	TLS_JOB_POOL = (Job*)(jobPoolBufferAligned)+TLS_WORKER_ID * jobContext->MAX_JOBS_PER_THREAD;
	return true;
}

bool KickJob(Job* job)
{
	return TLS_JOB_SYS_CONTEXT->mWorkerJobQueues[TLS_WORKER_ID]->Push(job);
}

static inline bool IsJobComplete(const Job* job)
{
	return job->unfinishedJobs == 0;
}

unsigned int GetWorkerId()
{
	return TLS_WORKER_ID;
}

void WaitForJob(Job* job)
{
	while (!IsJobComplete(job))
	{
		Job* nextJob = internal::GetValidJob();
		if (nextJob) { internal::ExecuteJob(nextJob); }
	}
}

void WaitUntilJobSystemTermination(JobSystemContext* jobSysContext)
{
	while (jobSysContext->mIsJobSystemActive)
	{
		Job* nextJob = internal::GetValidJob();
		if (nextJob) { internal::ExecuteJob(nextJob); }
	}
}

Job* CreateJob(JobFunction function)
{
	Job* job = internal::AllocateJob();
	job->function = function;
	job->parent = nullptr;
	job->unfinishedJobs = 1;

	// TODO: Handle the parameters being in the padding
	job->data = nullptr;

	return job;
}

Job* CreateJobAsChild(Job* parent, JobFunction function)
{
	assert(parent); // "Parent Job is null"
	parent->unfinishedJobs++;

	Job* job = internal::AllocateJob();
	job->function = function;
	job->parent = parent;
	job->unfinishedJobs = 1;

	// TODO: Handle the parameters being in the padding
	job->data = nullptr;

	return job;
}

}



