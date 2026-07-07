import httpx
import json
import asyncio
import time
import sys

API_URL = "http://localhost:8000/api/v1"
PDF_PATH = "sample-data/sbi-insu-pdf.pdf"

async def main():
    print("1. Uploading PDF...")
    try:
        with open(PDF_PATH, "rb") as f:
            files = {"files": ("sbi-insu-pdf.pdf", f, "application/pdf")}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{API_URL}/documents", files=files)
                if resp.status_code != 201:
                    print(f"Upload failed: {resp.status_code} {resp.text}")
                    return
                data = resp.json()
                job_id = data["job_id"]
                print(f"   Success! Job ID: {job_id}")
                
                print("2. Polling for job completion...")
                for i in range(30):
                    status_resp = await client.get(f"{API_URL}/documents/{job_id}")
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        status = status_data["status"]
                        print(f"   Status: {status}")
                        if status == "completed":
                            break
                        elif status == "failed":
                            print("Job failed.")
                            return
                    await asyncio.sleep(5)
                
                print("3. Fetching Report...")
                report_resp = await client.get(f"{API_URL}/documents/{job_id}/report")
                print(json.dumps(report_resp.json(), indent=2))
                
                print("4. Testing Q&A Query...")
                query = "What is the name of this insurance?"
                query_resp = await client.post(
                    f"{API_URL}/documents/{job_id}/query", 
                    json={"query": query},
                    timeout=30.0
                )
                print("Answer:")
                print(json.dumps(query_resp.json(), indent=2))
    except Exception as e:
        print(f"E2E Test Error: {e}")
            
if __name__ == "__main__":
    asyncio.run(main())
