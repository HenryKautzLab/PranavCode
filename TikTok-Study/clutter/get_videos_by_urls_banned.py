import yt_dlp
import time
import random
from datetime import datetime
import json
import os


class SafeTikTokDownloader:
    def __init__(self, output_folder='tiktok_downloads'):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # Rate limiting configuration
        self.min_delay = 8  # Minimum seconds between downloads
        self.max_delay = 15  # Maximum seconds between downloads
        self.batch_size = 10  # Videos per batch
        self.batch_delay = 60  # Seconds to wait between batches

        self.ydl_opts = {
            'outtmpl': f'{output_folder}/%(uploader)s_%(id)s.%(ext)s',
            'format': 'best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # Important: Add user agent to avoid detection
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            # Retry settings
            'retries': 3,
            'fragment_retries': 3,
            'sleep_interval': 5,
            'max_sleep_interval': 10,
        }

    def download_with_rate_limit(self, urls, resume_from=0):
        """
        Download videos with built-in rate limiting

        Args:
            urls: List of TikTok URLs
            resume_from: Index to resume from (useful if interrupted)
        """
        downloaded = []
        failed = []

        total = len(urls)
        print(f"Total videos to download: {total}")
        print(f"Estimated time: {self._estimate_time(total)} minutes")
        print(f"Starting from index: {resume_from}\n")

        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            for i, url in enumerate(urls[resume_from:], start=resume_from):
                try:
                    print(f"[{i + 1}/{total}] Downloading: {url}")

                    # Download video
                    info = ydl.extract_info(url, download=True)

                    downloaded.append({
                        'index': i,
                        'url': url,
                        'title': info.get('title', 'Unknown'),
                        'author': info.get('uploader', 'Unknown'),
                        'timestamp': datetime.now().isoformat()
                    })

                    print(f"✓ Success: {info.get('title', 'Unknown')}")

                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Failed: {error_msg}")

                    failed.append({
                        'index': i,
                        'url': url,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Check if it's a rate limit error
                    if '403' in error_msg or 'rate' in error_msg.lower():
                        print("\n⚠ WARNING: Possible rate limit detected!")
                        print("Waiting 5 minutes before continuing...")
                        time.sleep(300)  # Wait 5 minutes

                # Random delay between downloads
                delay = random.uniform(self.min_delay, self.max_delay)
                print(f"Waiting {delay:.1f} seconds...\n")
                time.sleep(delay)

                # Batch delay
                if (i + 1) % self.batch_size == 0 and i + 1 < total:
                    print(f"\n{'=' * 50}")
                    print(f"Completed batch of {self.batch_size} videos")
                    print(f"Taking a {self.batch_delay} second break...")
                    print(f"{'=' * 50}\n")
                    time.sleep(self.batch_delay)

                # Save progress after each download
                self._save_progress(i, downloaded, failed)

        self._save_final_report(downloaded, failed)
        return downloaded, failed

    def _estimate_time(self, total_videos):
        """Estimate total download time"""
        avg_delay = (self.min_delay + self.max_delay) / 2
        batch_delays = (total_videos // self.batch_size) * self.batch_delay
        total_seconds = (total_videos * avg_delay) + batch_delays
        return round(total_seconds / 60, 1)

    def _save_progress(self, index, downloaded, failed):
        """Save progress to resume later if interrupted"""
        progress = {
            'last_index': index,
            'downloaded': len(downloaded),
            'failed': len(failed),
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(self.output_folder, 'progress.json'), 'w') as f:
            json.dump(progress, f, indent=2)

    def _save_final_report(self, downloaded, failed):
        """Save final download report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total': len(downloaded) + len(failed),
            'successful': len(downloaded),
            'failed': len(failed),
            'success_rate': f"{(len(downloaded) / (len(downloaded) + len(failed)) * 100):.1f}%",
            'downloaded': downloaded,
            'failed': failed
        }

        report_file = os.path.join(self.output_folder, 'final_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total videos: {report['total']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
        print(f"Success rate: {report['success_rate']}")
        print(f"Report saved to: {report_file}")
        print(f"{'=' * 60}")

    def load_urls_from_file(self, filename):
        """Load URLs from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            if filename.endswith('.json'):
                data = json.load(f)
                # Adjust based on your JSON structure
                if isinstance(data, list):
                    return [item if isinstance(item, str) else item.get('url') for item in data]
            else:
                return [line.strip() for line in f if line.strip() and 'tiktok.com' in line]

    def resume_download(self):
        """Resume from last saved progress"""
        progress_file = os.path.join(self.output_folder, 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                return progress['last_index'] + 1
        return 0


# Usage
if __name__ == "__main__":
    downloader = SafeTikTokDownloader()

    # Load URLs
    urls = downloader.load_urls_from_file('tiktok_urls.txt')

    # Check if resuming
    resume_from = downloader.resume_download()
    if resume_from > 0:
        print(f"Resuming from video #{resume_from + 1}")

    # Download with rate limiting
    downloader.download_with_rate_limit(urls, resume_from=resume_from)