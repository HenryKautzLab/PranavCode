import json
import os
import yt_dlp
import time
import random
from datetime import datetime
from pathlib import Path


class TikTokDownloaderWithTimestamps:
    def __init__(self, output_folder='tiktok_downloads'):
        self.output_folder = output_folder
        self.json_file = None
        self.videos_data = []

        # Create output folders
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'metadata'), exist_ok=True)

        # Rate limiting settings
        self.min_delay = 8
        self.max_delay = 15
        self.batch_size = 10
        self.batch_delay = 60

        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'format': 'best',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            'retries': 3,
            'fragment_retries': 3,
        }

    def find_json_file(self):
        """Automatically find TikTok JSON file in current directory"""
        current_dir = Path('.')

        # Look for common TikTok export filenames
        possible_names = [
            'user_data.json',
            'tiktok_data.json',
            'data.json',
            'user_data_tiktok.json'
        ]

        # First check for exact matches
        for filename in possible_names:
            if (current_dir / filename).exists():
                self.json_file = filename
                print(f"✓ Found TikTok data file: {filename}")
                return True

        # Then search for any JSON file in current directory
        json_files = list(current_dir.glob('*.json'))

        if len(json_files) == 1:
            self.json_file = str(json_files[0])
            print(f"✓ Found JSON file: {self.json_file}")
            return True
        elif len(json_files) > 1:
            print("Multiple JSON files found. Please select one:")
            for i, file in enumerate(json_files, 1):
                print(f"  {i}. {file.name}")

            choice = input("\nEnter number (or press Enter to use user_data.json): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(json_files):
                self.json_file = str(json_files[int(choice) - 1])
                print(f"✓ Selected: {self.json_file}")
                return True
            else:
                self.json_file = 'user_data.json'
                if Path(self.json_file).exists():
                    return True

        print("✗ No TikTok JSON file found in current directory")
        print("Please ensure your TikTok data export JSON file is in the same folder")
        return False

    def load_watch_history(self):
        """Load watch history with timestamps from JSON file"""
        if not self.json_file:
            if not self.find_json_file():
                return False

        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract watch history
            watch_history = data.get("Your Activity", {}).get("Watch History", {})
            video_list = watch_history.get("VideoList", [])

            if not video_list:
                print("✗ No videos found in Watch History")
                return False

            # Store videos with their timestamps
            for video in video_list:
                url = video.get("Link", "")
                date_str = video.get("Date", "")

                if url:
                    # Parse the date
                    try:
                        watch_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    except:
                        watch_date = None

                    self.videos_data.append({
                        'url': url,
                        'watched_date': date_str,
                        'watched_datetime': watch_date,
                        'video_id': self._extract_video_id(url)
                    })

            # Sort by date (oldest first)
            self.videos_data.sort(key=lambda x: x['watched_datetime'] if x['watched_datetime'] else datetime.min)

            print(f"✓ Loaded {len(self.videos_data)} videos from watch history")
            self._print_date_summary()

            return True

        except Exception as e:
            print(f"✗ Error loading JSON file: {str(e)}")
            return False

    def _extract_video_id(self, url):
        """Extract video ID from TikTok URL"""
        try:
            # URL format: https://www.tiktokv.com/share/video/7465220866134117675/
            parts = url.rstrip('/').split('/')
            return parts[-1]
        except:
            return "unknown"

    def _print_date_summary(self):
        """Print summary of date range"""
        if self.videos_data:
            dates = [v['watched_datetime'] for v in self.videos_data if v['watched_datetime']]
            if dates:
                print(f"  Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    def _sanitize_date_for_filename(self, date_str):
        """Convert date string to filename-safe format"""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y%m%d_%H%M%S")
        except:
            return "unknown_date"

    def download_all(self):
        """Download all videos with timestamps preserved"""
        if not self.videos_data:
            print("✗ No videos to download. Load watch history first.")
            return

        total = len(self.videos_data)
        downloaded = []
        failed = []

        print(f"\n{'=' * 60}")
        print(f"Starting download of {total} videos")
        print(f"Estimated time: {self._estimate_time(total)} minutes")
        print(f"{'=' * 60}\n")

        for i, video_data in enumerate(self.videos_data, 1):
            url = video_data['url']
            watched_date = video_data['watched_date']
            date_prefix = self._sanitize_date_for_filename(watched_date)
            video_id = video_data['video_id']

            # Update output template with date prefix
            self.ydl_opts['outtmpl'] = os.path.join(
                self.output_folder,
                'videos',
                f'{date_prefix}_{video_id}_%(uploader)s.%(ext)s'
            )

            try:
                print(f"[{i}/{total}] Watched: {watched_date}")
                print(f"         Downloading: {url}")

                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)

                download_record = {
                    'index': i,
                    'url': url,
                    'video_id': video_id,
                    'watched_date': watched_date,
                    'downloaded_date': datetime.now().isoformat(),
                    'title': info.get('title', 'Unknown'),
                    'author': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'filename': f'{date_prefix}_{video_id}_{info.get("uploader", "unknown")}.{info.get("ext", "mp4")}'
                }

                downloaded.append(download_record)
                print(f"✓ Success: {info.get('title', 'Unknown')}\n")

                # Save individual metadata
                self._save_video_metadata(download_record)

            except Exception as e:
                error_msg = str(e)
                print(f"✗ Failed: {error_msg}\n")

                failed.append({
                    'index': i,
                    'url': url,
                    'video_id': video_id,
                    'watched_date': watched_date,
                    'error': error_msg,
                    'failed_date': datetime.now().isoformat()
                })

                # Check for rate limiting
                if '403' in error_msg or 'rate' in error_msg.lower():
                    print("⚠ Rate limit detected! Waiting 5 minutes...")
                    time.sleep(300)

            # Progress save
            self._save_progress(i, downloaded, failed)

            # Rate limiting delay
            if i < total:
                delay = random.uniform(self.min_delay, self.max_delay)
                print(f"⏳ Waiting {delay:.1f} seconds...\n")
                time.sleep(delay)

            # Batch delay
            if i % self.batch_size == 0 and i < total:
                print(f"\n{'=' * 60}")
                print(f"Completed batch of {self.batch_size} videos")
                print(f"Taking a {self.batch_delay} second break...")
                print(f"{'=' * 60}\n")
                time.sleep(self.batch_delay)

        # Save final report
        self._save_final_report(downloaded, failed)

        return downloaded, failed

    def _save_video_metadata(self, metadata):
        """Save metadata for each video"""
        metadata_file = os.path.join(
            self.output_folder,
            'metadata',
            f'{metadata["video_id"]}_metadata.json'
        )

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _save_progress(self, index, downloaded, failed):
        """Save progress checkpoint"""
        progress = {
            'last_index': index,
            'downloaded': len(downloaded),
            'failed': len(failed),
            'timestamp': datetime.now().isoformat(),
            'downloaded_list': downloaded,
            'failed_list': failed
        }

        progress_file = os.path.join(self.output_folder, 'progress.json')
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def _save_final_report(self, downloaded, failed):
        """Save comprehensive final report"""
        report = {
            'summary': {
                'total_videos': len(self.videos_data),
                'downloaded': len(downloaded),
                'failed': len(failed),
                'success_rate': f"{(len(downloaded) / len(self.videos_data) * 100):.1f}%",
                'report_generated': datetime.now().isoformat()
            },
            'date_range': {
                'oldest_watched': self.videos_data[0]['watched_date'] if self.videos_data else None,
                'newest_watched': self.videos_data[-1]['watched_date'] if self.videos_data else None
            },
            'downloaded_videos': downloaded,
            'failed_videos': failed
        }

        # Save detailed JSON report
        report_file = os.path.join(self.output_folder, 'download_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save CSV for easy viewing
        self._save_csv_report(downloaded, failed)

        # Print summary
        self._print_final_summary(report)

    def _save_csv_report(self, downloaded, failed):
        """Save a CSV file with download information"""
        import csv

        csv_file = os.path.join(self.output_folder, 'download_report.csv')

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Status', 'Index', 'Watched Date', 'Video ID',
                'Title', 'Author', 'Duration', 'Views', 'Filename', 'URL'
            ])

            for video in downloaded:
                writer.writerow([
                    'SUCCESS',
                    video['index'],
                    video['watched_date'],
                    video['video_id'],
                    video['title'],
                    video['author'],
                    video['duration'],
                    video['view_count'],
                    video['filename'],
                    video['url']
                ])

            for video in failed:
                writer.writerow([
                    'FAILED',
                    video['index'],
                    video['watched_date'],
                    video['video_id'],
                    '-',
                    '-',
                    '-',
                    '-',
                    '-',
                    video['url']
                ])

        print(f"✓ CSV report saved: {csv_file}")

    def _print_final_summary(self, report):
        """Print final summary"""
        print(f"\n{'=' * 60}")
        print("DOWNLOAD COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total videos: {report['summary']['total_videos']}")
        print(f"Downloaded: {report['summary']['downloaded']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success rate: {report['summary']['success_rate']}")
        print(f"\nDate range:")
        print(f"  Oldest: {report['date_range']['oldest_watched']}")
        print(f"  Newest: {report['date_range']['newest_watched']}")
        print(f"\nFiles saved in: {self.output_folder}/")
        print(f"  - Videos: {self.output_folder}/videos/")
        print(f"  - Metadata: {self.output_folder}/metadata/")
        print(f"  - Reports: {self.output_folder}/download_report.json")
        print(f"            {self.output_folder}/download_report.csv")
        print(f"{'=' * 60}\n")

    def _estimate_time(self, total):
        """Estimate total download time in minutes"""
        avg_delay = (self.min_delay + self.max_delay) / 2
        batch_delays = (total // self.batch_size) * self.batch_delay
        total_seconds = (total * avg_delay) + batch_delays
        return round(total_seconds / 60, 1)


def main():
    print("=" * 60)
    print("TikTok Watch History Downloader")
    print("(With Original Timestamps)")
    print("=" * 60 + "\n")

    # Initialize downloader
    downloader = TikTokDownloaderWithTimestamps()

    # Load watch history from JSON
    if not downloader.load_watch_history():
        print("\nExiting...")
        return

    # Confirm before downloading
    print(f"\nReady to download {len(downloader.videos_data)} videos")
    proceed = input("Proceed with download? (y/n): ").strip().lower()

    if proceed != 'y':
        print("Download cancelled.")
        return

    # Start download
    print("\n" + "=" * 60)
    downloader.download_all()

    print("\n✓ All done! Check the output folder for your videos.")


if __name__ == "__main__":
    main()
