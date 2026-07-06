import asyncio
import argparse
import time
import random
import logging
from typing import List, Dict, Any
import httpx
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("stress_test")


def generate_mock_pdf(kb_size: int, filename: str) -> bytes:
    """Generates a dummy PDF file of a specified approximate size in KB."""
    doc = fitz.open()
    page = doc.new_page()
    
    # Generate content to match size requirement roughly
    # A single page can hold some text. Let's write paragraphs.
    words = ["dotkonnekt", "scalability", "architecture", "ingestion", "vector", "database", "concurrency", "performance"]
    paragraphs = []
    
    # Roughly 1 KB is ~150 words. Let's repeat to get target size.
    approx_words_needed = int(kb_size * 150)
    text_content = " ".join(random.choices(words, k=approx_words_needed))
    
    # Write text to the page
    rect = fitz.Rect(50, 50, 550, 750)
    page.insert_textbox(rect, text_content, fontsize=9)
    
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


async def run_single_job(
    client: httpx.AsyncClient,
    base_url: str,
    pdf_bytes: bytes,
    filename: str,
    job_index: int,
) -> Dict[str, Any]:
    start_time = time.time()
    
    # 1. Upload files
    files = {"files": (filename, pdf_bytes, "application/pdf")}
    data = {"collection": "stress_test_collection", "metadata": '{"source": "stress_test"}'}
    
    upload_url = f"{base_url}/api/v1/documents"
    
    try:
        response = await client.post(upload_url, files=files, data=data, timeout=30.0)
        if response.status_code != 201:
            logger.error(f"Job {job_index}: Upload failed with status {response.status_code}: {response.text}")
            return {"status": "failed", "error": f"Upload status {response.status_code}", "duration": 0}
            
        res_data = response.json()
        job_id = res_data["job_id"]
        logger.info(f"Job {job_index}: Uploaded successfully. Job ID: {job_id}")
        
        # 2. Poll job status
        poll_url = f"{base_url}/api/v1/documents/{job_id}"
        attempts = 0
        max_attempts = 60
        
        while attempts < max_attempts:
            await asyncio.sleep(1.0)
            attempts += 1
            
            poll_resp = await client.get(poll_url, timeout=10.0)
            if poll_resp.status_code == 200:
                poll_data = poll_resp.json()
                status = poll_data["status"]
                
                if status == "completed":
                    duration = time.time() - start_time
                    logger.info(f"Job {job_index}: Completed in {duration:.2f}s")
                    return {
                        "status": "completed",
                        "duration": duration,
                        "size_bytes": len(pdf_bytes),
                    }
                elif status == "failed":
                    logger.error(f"Job {job_index}: Pipeline reported failure")
                    return {"status": "failed", "error": "Ingestion pipeline error", "duration": 0}
            else:
                logger.warning(f"Job {job_index}: Polling failed with status {poll_resp.status_code}")
                
        return {"status": "timeout", "error": "Polling timeout exceeded", "duration": 0}
        
    except Exception as e:
        logger.exception(f"Job {job_index}: Exception occurred during lifecycle")
        return {"status": "failed", "error": str(e), "duration": 0}


async def stress_test_runner(
    base_url: str,
    concurrency: int,
    num_jobs: int,
    file_size_kb: int,
):
    logger.info("Initializing stress test...")
    logger.info(f"Parameters: Base URL={base_url}, Concurrency={concurrency}, Num Jobs={num_jobs}, Avg File Size={file_size_kb}KB")
    
    # Pre-generate mock PDF bytes to avoid CPU time during test
    logger.info("Pre-generating mock PDF files...")
    mock_pdfs = [
        (generate_mock_pdf(file_size_kb, f"doc_{i}.pdf"), f"doc_{i}.pdf")
        for i in range(num_jobs)
    ]
    
    sem = asyncio.Semaphore(concurrency)
    results = []
    
    async with httpx.AsyncClient() as client:
        # Check health
        try:
            health_resp = await client.get(f"{base_url}/health")
            logger.info(f"Service health check: {health_resp.status_code} - {health_resp.json()}")
        except Exception as e:
            logger.error(f"Cannot connect to server at {base_url}/health: {e}")
            return
            
        start_time = time.time()
        
        async def worker(pdf_bytes: bytes, filename: str, idx: int):
            async with sem:
                res = await run_single_job(client, base_url, pdf_bytes, filename, idx)
                results.append(res)
                
        tasks = [
            worker(pdf_bytes, filename, i)
            for i, (pdf_bytes, filename) in enumerate(mock_pdfs)
        ]
        
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
    # Analyze results
    successes = [r for r in results if r["status"] == "completed"]
    failures = [r for r in results if r["status"] in ("failed", "timeout")]
    
    total_bytes = sum(s["size_bytes"] for s in successes)
    durations = [s["duration"] for s in successes]
    avg_latency = sum(durations) / len(durations) if durations else 0
    max_latency = max(durations) if durations else 0
    min_latency = min(durations) if durations else 0
    
    print("\n" + "="*50)
    print("               STRESS TEST RESULTS")
    print("="*50)
    print(f"Total Test Time:      {total_time:.2f} seconds")
    print(f"Total Jobs Dispatched: {num_jobs}")
    print(f"Successful Ingestions: {len(successes)} ({len(successes)/num_jobs*100:.1f}%)")
    print(f"Failed Ingestions:     {len(failures)} ({len(failures)/num_jobs*100:.1f}%)")
    print(f"Throughput:            {len(successes)/total_time:.2f} jobs/second")
    print(f"Data Throughput:       {(total_bytes / 1024 / 1024) / total_time:.2f} MB/second")
    print(f"Total Data Ingested:   {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Average Job Latency:   {avg_latency:.2f} seconds")
    print(f"Min / Max Latency:     {min_latency:.2f}s / {max_latency:.2f}s")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="dotkonnekt RAG Ingestion Stress Test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent uploads")
    parser.add_argument("--num-jobs", type=int, default=20, help="Total number of jobs to run")
    parser.add_argument("--size-kb", type=int, default=100, help="Approximate file size of each PDF in KB")
    
    args = parser.parse_args()
    
    asyncio.run(
        stress_test_runner(
            base_url=args.url,
            concurrency=args.concurrency,
            num_jobs=args.num_jobs,
            file_size_kb=args.size_kb
        )
    )


if __name__ == "__main__":
    main()
