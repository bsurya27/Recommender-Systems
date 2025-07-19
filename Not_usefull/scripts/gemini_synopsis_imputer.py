import pandas as pd
import os
import re
import glob
import time
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ------------------------ SETUP ------------------------
def setup_openai(api_key):
    """Setup OpenAI client."""
    return OpenAI(api_key=api_key)

def setup_synopsis_directory():
    """Create directory for storing individual synopsis files"""
    synopsis_dir = Path('generated_synopses')
    synopsis_dir.mkdir(exist_ok=True)
    return synopsis_dir


def setup_logging():
    log_filename = f'synopsis_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename


# ------------------------ DATA HANDLING ------------------------
def read_wiki_page(anime_id):
    """Read the wiki page content and extract the Plot section if available"""
    wiki_file = Path(f'dataset/archive/wiki_pages/wiki_pages/{anime_id}.txt')
    if wiki_file.exists():
        with open(wiki_file, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'==\s*Plot\s*==([\s\S]*?)(==|$)', content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return content.strip()  # fallback to full content
    return None


def get_processed_anime_ids_from_logs():
    processed_ids = set()
    for log_file in glob.glob('synopsis_generation_*.log'):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ('✓ Generated synopsis for' in line or '✗ Failed to generate synopsis for' in line) and 'ID:' in line:
                        anime_id = line.split('ID:')[-1].split(')')[0].strip()
                        processed_ids.add(int(anime_id))
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
    return processed_ids

def get_processed_anime_ids_from_files(synopsis_dir):
    """Get processed anime IDs from existing synopsis files"""
    processed_ids = set()
    for file_path in synopsis_dir.glob('*.txt'):
        try:
            anime_id = int(file_path.stem)  # filename without extension
            processed_ids.add(anime_id)
        except ValueError:
            continue  # skip files that don't have numeric names
    return processed_ids


# ------------------------ GENERATION ------------------------
def generate_synopsis_openai(client, anime_name, wiki_content, english_name=None):
    anime_name = (anime_name or '').strip()
    english_name = (english_name or '').strip()

    if anime_name and english_name and english_name.lower() != anime_name.lower():
        name_info = f'"{anime_name}" (English: "{english_name}")'
    elif anime_name:
        name_info = f'"{anime_name}"'
    elif english_name:
        name_info = f'"{english_name}"'
    else:
        name_info = '"Unknown Anime"'

    prompt = f"""
You are an expert anime reviewer. Based on the Wikipedia plot below, write a short, spoiler-free synopsis for the anime {name_info}. 
Before generating the synopsis, check if the name is correct. 
The synopsis should:
- Be short and concise
- Be engaging and informative
- Avoid spoilers for major plot points
- Focus on the plot or premise
- Be written in English

Wrap the synopsis in [SYNOPSIS] and [/SYNOPSIS] tags.

Wikipedia content:
{wiki_content}

Output:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.0-mini",
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error for {anime_name}: {e}")
        return None


# ------------------------ MAIN ------------------------
def main():
    log_filename = setup_logging()
    logger = logging.getLogger(__name__)

    # Setup synopsis directory
    synopsis_dir = setup_synopsis_directory()
    logger.info(f"Synopsis directory: {synopsis_dir}")

    # Get processed IDs from both logs and existing files
    processed_ids_logs = get_processed_anime_ids_from_logs()
    processed_ids_files = get_processed_anime_ids_from_files(synopsis_dir)
    processed_ids = processed_ids_logs.union(processed_ids_files)
    
    logger.info(f"Resuming... Found {len(processed_ids)} processed anime IDs.")

    anime_df = pd.read_csv('dataset/Anime.csv')
    missing_df = anime_df[anime_df['synopsis'].isna() | anime_df['synopsis'].str.strip().eq('')]

    logger.info(f"Total missing synopses: {len(missing_df)}")

    missing_with_wiki = []
    for _, row in missing_df.iterrows():
        anime_id = row['anime_id']
        if anime_id in processed_ids:
            continue
        wiki = read_wiki_page(anime_id)
        if wiki:
            row_dict = row.to_dict()
            row_dict['wiki_content'] = wiki
            missing_with_wiki.append(row_dict)

    if not missing_with_wiki:
        logger.info("No new entries with wiki content found.")
        return

    logger.info(f"Found {len(missing_with_wiki)} animes to process")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not found")
        return

    client = setup_openai(api_key)

    for idx, anime_data in enumerate(missing_with_wiki):
        anime_id = anime_data['anime_id']
        name = anime_data['name']
        eng_name = anime_data.get('english_name')
        wiki_content = anime_data['wiki_content']

        logger.info(f"Generating for: {name} (ID: {anime_id})")

        output = generate_synopsis_openai(client, name, wiki_content, eng_name)
        if output:
            # Save synopsis to individual text file
            synopsis_file = synopsis_dir / f"{anime_id}.txt"
            try:
                with open(synopsis_file, 'w', encoding='utf-8') as f:
                    f.write(f"Anime ID: {anime_id}\n")
                    f.write(f"Name: {name}\n")
                    if eng_name:
                        f.write(f"English Name: {eng_name}\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 50 + "\n")
                    f.write(output)
                logger.info(f"✓ Generated synopsis for {name}: {output[:100]}... (ID: {anime_id})")
            except Exception as e:
                logger.error(f"Error saving synopsis file for {name} (ID: {anime_id}): {e}")
        else:
            logger.error(f"✗ Failed to generate synopsis for {name} (ID: {anime_id})")

        logger.info(f"Progress: {idx+1}/{len(missing_with_wiki)}")

        time.sleep(4)

    logger.info("All done.")


if __name__ == "__main__":
    main()
