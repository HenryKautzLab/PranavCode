# RESEARCH-COMPLIANT BULK DOWNLOADER
# For academic use with proper approvals

import json
import yt_dlp
import time
import random
import logging
from datetime import datetime
import sqlite3


class ResearchCompliantDownloader:
    def __init__(self,
                 project_name,
                 irb_number=None,
                 rate_limit_per_hour=30):
        """
        project_name: Your research project name
        irb_number: IRB approval number (if applicable)
        rate_limit_per_hour: Conservative rate limiting
        """
        self.project_name = project_name
        self.irb_number = irb_number
        self.rate_limit = rate_limit_per_hour

        # Set up logging for documentation
        logging.basicConfig(
            filename=f'download_log_{project_name}.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.init_database()

        # Conservative yt-dlp settings
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Don't need 4K for research
            'outtmpl': f'videos/{project_name}/%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'retries': 3,
            'fragment_retries': 3,
            'socket_timeout': 30,
            # Rate limiting built into yt-dlp
            'sleep_interval': 5,
            'max_sleep_interval': 15,
        }

        logging.info(f"Initialized downloader for project: {project_name}")
        if irb_number:
            logging.info(f"IRB Approval: {irb_number}")

    def init_database(self):
        """Track all downloads for research documentation"""
        conn = sqlite3.connect('research_downloads.db')
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS downloads
                     (video_id TEXT PRIMARY KEY,
                      url TEXT,
                      original_url TEXT,
                      watched_date TEXT,
                      download_date TEXT,
                      download_success BOOLEAN,
                      file_path TEXT,
                      file_size INTEGER,
                      error_message TEXT,
                      project_name TEXT,
                      irb_number TEXT)''')

        conn.commit()
        conn.close()

    def load_urls_from_export(self, json_file):
        """Load URLs from TikTok data export"""
        with open(json_file, 'r') as f:
            data = json.load(f)

        videos = data['Your Activity']['Watch History']['VideoList']

        logging.info(f"Loaded {len(videos)} videos from {json_file}")
        return videos

    def convert_url(self, url):
        """Convert tiktokv.com to tiktok.com"""
        import re
        match = re.search(r'video/(\d+)', url)
        if match and 'tiktokv.com' in url:
            video_id = match.group(1)
            return f"https://www.tiktok.com/@/video/{video_id}"
        return url

    def download_batch(self, videos, start_index=0, batch_size=100):
        """
        Download a batch of videos with proper rate limiting

        Args:
            videos: List of video data from TikTok export
            start_index: Where to start in the list
            batch_size: How many to download in this session
        """
        end_index = min(start_index + batch_size, len(videos))
        batch = videos[start_index:end_index]

        print(f"\n{'=' * 60}")
        print(f"Downloading videos {start_index + 1} to {end_index}")
        print(f"Project: {self.project_name}")
        if self.irb_number:
            print(f"IRB: {self.irb_number}")
        print(f"{'=' * 60}\n")

        downloaded = 0
        failed = 0

        for i, video in enumerate(batch, start=start_index + 1):
            original_url = video['Link']
            watched_date = video['Date']

            # Convert URL
            converted_url = self.convert_url(original_url)
            video_id = converted_url.split('/')[-1]

            print(f"[{i}/{len(videos)}] Processing: {video_id}")
            print(f"  Watched: {watched_date}")

            try:
                # Download with yt-dlp
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    info = ydl.extract_info(converted_url, download=True)

                    # Get file info
                    file_path = ydl.prepare_filename(info)
                    import os
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

                    # Log to database
                    self.log_download(
                        video_id=video_id,
                        url=converted_url,
                        original_url=original_url,
                        watched_date=watched_date,
                        success=True,
                        file_path=file_path,
                        file_size=file_size
                    )

                    print(f"  ✓ Downloaded: {info.get('title', 'Unknown')}")
                    downloaded += 1

            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Failed: {error_msg}")

                # Log failure
                self.log_download(
                    video_id=video_id,
                    url=converted_url,
                    original_url=original_url,
                    watched_date=watched_date,
                    success=False,
                    error_message=error_msg
                )

                failed += 1

                # If rate limited, stop for today
                if '403' in error_msg or '429' in error_msg:
                    print("\n⚠️  Rate limit detected. Stopping for today.")
                    print("Resume tomorrow by setting start_index={i}")
                    break

            # Rate limiting: seconds per video based on hourly limit
            seconds_per_video = 3600 / self.rate_limit
            delay = random.uniform(seconds_per_video * 0.8, seconds_per_video * 1.2)

            if i < end_index:
                print(f"  ⏳ Waiting {delay:.0f}s...\n")
                time.sleep(delay)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"Batch Complete")
        print(f"Downloaded: {downloaded}")
        print(f"Failed: {failed}")
        print(f"Next start_index: {end_index}")
        print(f"{'=' * 60}\n")

        logging.info(f"Batch complete: {downloaded} downloaded, {failed} failed")

    def log_download(self, video_id, url, original_url, watched_date,
                     success, file_path=None, file_size=None, error_message=None):
        """Log download to database"""
        conn = sqlite3.connect('research_downloads.db')
        c = conn.cursor()

        c.execute('''INSERT OR REPLACE INTO downloads 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (video_id, url, original_url, watched_date,
                   datetime.now().isoformat(), success, file_path,
                   file_size, error_message, self.project_name, self.irb_number))

        conn.commit()
        conn.close()

        logging.info(f"{'SUCCESS' if success else 'FAILED'}: {video_id}")

    def generate_research_report(self):
        """Generate documentation for research compliance"""
        conn = sqlite3.connect('research_downloads.db')
        c = conn.cursor()

        c.execute('''SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN download_success THEN 1 ELSE 0 END) as successful,
                        SUM(file_size) as total_size
                     FROM downloads 
                     WHERE project_name = ?''',
                  (self.project_name,))

        stats = c.fetchone()
        conn.close()

        report = f"""
Research Data Collection Report
{'=' * 60}
Project: {self.project_name}
IRB Number: {self.irb_number or 'N/A'}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Collection Statistics:
- Total videos attempted: {stats[0]}
- Successfully downloaded: {stats[1]}
- Total data size: {stats[2] / (1024 ** 3):.2f} GB

Collection Method:
- Tool: yt-dlp
- Rate limiting: {self.rate_limit} videos/hour
- Source: TikTok personal data export
- Compliance: Conservative rate limiting applied

Notes:
- All downloads logged in research_downloads.db
- Video IDs and metadata preserved for replicability
- Original watch dates maintained for temporal analysis
{'=' * 60}
        """

        print(report)

        # Save report
        with open(f'research_report_{self.project_name}.txt', 'w') as f:
            f.write(report)

        return report


# USAGE FOR RESEARCH
if __name__ == "__main__":
    # Initialize with your research details
    downloader = ResearchCompliantDownloader(
        project_name="tiktok_content_analysis_2025",
        irb_number="IRB-2025-001",  # Your IRB number
        rate_limit_per_hour=30  # Conservative: 30 per hour = 720/day
    )

    # Load your TikTok data export
    videos = downloader.load_urls_from_export('user_data_tiktok.json')

    # Download in daily batches
    # Day 1: Download first 100
    downloader.download_batch(videos, start_index=0, batch_size=100)

    # Day 2: Download next 100
    # downloader.download_batch(videos, start_index=100, batch_size=100)

    # Generate documentation
    downloader.generate_research_report()