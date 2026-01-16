import json
import yt_dlp
import time
import random
import logging
from datetime import datetime
import sqlite3
import os


class ResearchCompliantDownloader:
    def __init__(self,
                 project_name,
                 irb_number=None,
                 rate_limit_per_hour=30,
                 timeout_per_video=30):

        self.project_name = project_name
        self.irb_number = irb_number
        self.rate_limit = rate_limit_per_hour
        self.timeout = timeout_per_video

        os.makedirs(f'videos/{project_name}', exist_ok=True)

        logging.basicConfig(
            filename=f'download_log_{project_name}.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.init_database()

        self.ydl_opts = {
            'format': 'best/bestvideo+bestaudio/bestvideo/bestaudio',
            'outtmpl': f'videos/{project_name}/%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'retries': 0,  # CHANGED: No retries - fail immediately
            'fragment_retries': 0,  # CHANGED: No fragment retries
            'socket_timeout': 10,  # CHANGED: 10 seconds max, not 15
            'sleep_interval': 0,  # CHANGED: No sleep between retries
            'max_sleep_interval': 0,  # CHANGED: No sleep
            'ignoreerrors': False,
        }

        logging.info(f"Initialized downloader for project: {project_name}")

    def init_database(self):
        """Fixed database schema"""
        conn = sqlite3.connect('research_downloads.db')
        c = conn.cursor()

        c.execute('DROP TABLE IF EXISTS downloads')

        c.execute('''CREATE TABLE downloads
                     (video_id TEXT PRIMARY KEY,
                      url TEXT,
                      original_url TEXT,
                      watched_date TEXT,
                      download_date TEXT,
                      download_success INTEGER,
                      file_path TEXT,
                      file_size INTEGER,
                      error_message TEXT,
                      error_type TEXT,
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

    def is_video_unavailable_error(self, error_msg):
        """Detect if video is unavailable - FIXED to not confuse with IP block"""
        unavailable_indicators = [
            'video is unavailable',
            'video not available',
            'video not found',
            'this video is private',
            'account is private',
            'page doesn\'t exist',
            'content is no longer available',
            'removed',
            'deleted',
            '404',
            'http error 404',
            'is blocked from accessing this post',  # This is video unavailable, not IP block!
        ]

        error_lower = error_msg.lower()
        return any(indicator.lower() in error_lower for indicator in unavailable_indicators)

    def quick_check_video(self, url):
        """Quick check - FASTER"""
        try:
            quick_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'socket_timeout': 5,  # CHANGED: 5 seconds only
                'retries': 0,  # CHANGED: No retries
            }

            with yt_dlp.YoutubeDL(quick_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return True, None

        except Exception as e:
            error_msg = str(e)

            if self.is_video_unavailable_error(error_msg):
                return False, 'unavailable'

            return False, error_msg

    def download_single_video(self, url, video_id):
        """Download single video - FIXED to catch format errors"""
        try:
            # Remove ignoreerrors so we can catch the actual error
            download_opts = self.ydl_opts.copy()
            download_opts['ignoreerrors'] = False  # We want to see the errors!

            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is None:
                    return False, None, None, 'unavailable'

                file_path = ydl.prepare_filename(info)

                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    return True, file_path, file_size, None
                else:
                    return False, None, None, 'download_failed'

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)

            # Handle format errors by trying again with any format
            if 'Requested format is not available' in error_msg or 'format' in error_msg.lower():
                try:
                    # Try again with absolutely any format available
                    fallback_opts = self.ydl_opts.copy()
                    fallback_opts['format'] = 'best/bestvideo+bestaudio/bestvideo/bestaudio'
                    fallback_opts['ignoreerrors'] = False

                    with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                        info = ydl.extract_info(url, download=True)

                        if info is None:
                            return False, None, None, 'unavailable'

                        file_path = ydl.prepare_filename(info)

                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            return True, file_path, file_size, None
                        else:
                            return False, None, None, 'download_failed'

                except Exception as e2:
                    error_msg = str(e2)

            if self.is_video_unavailable_error(error_msg):
                return False, None, None, 'unavailable'

            return False, None, None, error_msg

        except Exception as e:
            error_msg = str(e)

            if self.is_video_unavailable_error(error_msg):
                return False, None, None, 'unavailable'

            return False, None, None, error_msg

    def download_batch(self, videos, start_index=0, batch_size=100):
        """Download batch"""
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
        unavailable = 0
        network_errors = 0

        for i, video in enumerate(batch, start=start_index + 1):
            original_url = video['Link']
            watched_date = video['Date']

            converted_url = self.convert_url(original_url)
            video_id = converted_url.split('/')[-1]

            print(f"[{i}/{len(videos)}] {video_id}")
            print(f"  Watched: {watched_date}")

            # Quick check
            print(f"  🔍 Checking...", end=' ', flush=True)
            available, check_error = self.quick_check_video(converted_url)

            if not available and check_error == 'unavailable':
                print(f"✗ UNAVAILABLE")
                unavailable += 1

                self.log_download(
                    video_id=video_id,
                    url=converted_url,
                    original_url=original_url,
                    watched_date=watched_date,
                    success=False,
                    error_message="Video unavailable",
                    error_type='unavailable'
                )

                time.sleep(1)  # CHANGED: Just 1 second
                continue

            print("✓")

            # Download
            print(f"  📥 Downloading...", end=' ', flush=True)

            success, file_path, file_size, error = self.download_single_video(
                converted_url, video_id
            )

            if success:
                print(f"✓ ({file_size / (1024 * 1024):.1f} MB)")
                downloaded += 1

                self.log_download(
                    video_id=video_id,
                    url=converted_url,
                    original_url=original_url,
                    watched_date=watched_date,
                    success=True,
                    file_path=file_path,
                    file_size=file_size,
                    error_type='success'
                )

                # Normal rate limiting for successful downloads
                seconds_per_video = 3600 / self.rate_limit
                delay = random.uniform(seconds_per_video * 0.8, seconds_per_video * 1.2)

                if i < end_index:
                    print(f"  ⏳ {delay:.0f}s\n")
                    time.sleep(delay)

            else:
                # Check if it's a network error
                is_network_error = (error and ('nodename' in error.lower() or
                                               'transport' in error.lower() or
                                               'dns' in error.lower() or
                                               'connection' in error.lower() or
                                               'network' in error.lower()))

                if error == 'unavailable':
                    print(f"✗ UNAVAILABLE")
                    unavailable += 1
                    error_type = 'unavailable'
                elif is_network_error:
                    print(f"✗ NETWORK ERROR")
                    network_errors += 1
                    error_type = 'network_error'
                else:
                    print(f"✗ ERROR: {error[:50]}")
                    failed += 1
                    error_type = 'error'

                self.log_download(
                    video_id=video_id,
                    url=converted_url,
                    original_url=original_url,
                    watched_date=watched_date,
                    success=False,
                    error_message=error,
                    error_type=error_type
                )

                # CHANGED: No delay for network errors or unavailable videos
                if is_network_error or error == 'unavailable':
                    print()
                else:
                    # Small delay for other errors
                    if i < end_index:
                        print(f"  ⏳ 5s\n")
                        time.sleep(5)

            # Progress
            if i % 5 == 0:
                print(f"\n  📊 {downloaded} ✓ | {unavailable} unavail | {network_errors} network | {failed} other\n")

            # Check if too many network errors in a row
            if network_errors > 3 and downloaded == 0:
                print(f"\n⚠️  Too many network errors. Check your internet connection!")
                print(f"⚠️  Network might be down or TikTok might be blocking DNS")
                break

        # Summary
        print(f"\n{'=' * 60}")
        print(f"✓ Batch Complete")
        print(f"{'=' * 60}")
        print(f"Downloaded:      {downloaded}")
        print(f"Unavailable:     {unavailable}")
        print(f"Network errors:  {network_errors}")
        print(f"Other errors:    {failed}")
        print(f"{'=' * 60}\n")

    def log_download(self, video_id, url, original_url, watched_date,
                     success, file_path=None, file_size=None,
                     error_message=None, error_type=None):
        """Log download to database"""
        conn = sqlite3.connect('research_downloads.db')
        c = conn.cursor()

        success_int = 1 if success else 0

        try:
            c.execute('''INSERT OR REPLACE INTO downloads 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (video_id, url, original_url, watched_date,
                       datetime.now().isoformat(), success_int, file_path,
                       file_size, error_message, error_type,
                       self.project_name, self.irb_number))

            conn.commit()
        except Exception as e:
            print(f"Database error: {e}")
            logging.error(f"Database error: {e}")
        finally:
            conn.close()

    def generate_research_report(self):
        """Generate report - FIXED zero division"""
        conn = sqlite3.connect('research_downloads.db')
        c = conn.cursor()

        c.execute('''SELECT 
                        error_type,
                        COUNT(*) as count
                     FROM downloads 
                     WHERE project_name = ?
                     GROUP BY error_type''',
                  (self.project_name,))

        stats = c.fetchall()

        c.execute('''SELECT SUM(file_size) 
                     FROM downloads 
                     WHERE project_name = ? AND download_success = 1''',
                  (self.project_name,))

        total_size = c.fetchone()[0] or 0

        conn.close()

        stats_dict = {row[0]: row[1] for row in stats}

        successful = stats_dict.get('success', 0)
        unavailable = stats_dict.get('unavailable', 0)
        error = stats_dict.get('error', 0)
        total = sum(stats_dict.values())

        # FIXED: Avoid zero division
        success_pct = (successful / total * 100) if total > 0 else 0
        unavail_pct = (unavailable / total * 100) if total > 0 else 0
        error_pct = (error / total * 100) if total > 0 else 0

        report = f"""
Research Data Collection Report
{'=' * 60}
Project: {self.project_name}
IRB: {self.irb_number or 'N/A'}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
- Total attempted:      {total}
- Downloaded:           {successful} ({success_pct:.1f}%)
- Unavailable:          {unavailable} ({unavail_pct:.1f}%)
- Other errors:         {error} ({error_pct:.1f}%)

Data Size: {total_size / (1024 ** 3):.2f} GB

Method: yt-dlp, {self.rate_limit} videos/hour
{'=' * 60}
        """

        print(report)

        with open(f'research_report_{self.project_name}.txt', 'w') as f:
            f.write(report)


# USAGE
if __name__ == "__main__":
    downloader = ResearchCompliantDownloader(
        project_name="tiktok_content_analysis_2025",
        irb_number="IRB-2025-001",
        rate_limit_per_hour=30,
        timeout_per_video=30
    )

    videos = downloader.load_urls_from_export('user_data_tiktok.json')

    downloader.download_batch(videos, start_index=0, batch_size=100)

    downloader.generate_research_report()