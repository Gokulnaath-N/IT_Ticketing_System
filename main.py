import os
import subprocess
import logging

os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/main_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_script(script_name: str) -> None:
    try:
        logging.info(f'Running {script_name}...')
        result = subprocess.run(['python', os.path.join('scripts', script_name)], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(result.stderr)
            raise RuntimeError(f'{script_name} failed')
        logging.info(f'{script_name} succeeded')
        print(f'{script_name} completed')
    except Exception as e:
        logging.error(f'Error in {script_name}: {e}')
        raise


def main():
    run_script('data_prep.py')
    run_script('feature_engineering.py')
    run_script('regression_model.py')
    run_script('clustering.py')
    logging.info('Pipeline completed. Launching Streamlit...')
    try:
        subprocess.call(['streamlit', 'run', 'scripts/dashboard.py'])
    except Exception as e:
        logging.error(f'Error launching Streamlit: {e}')


if __name__ == '__main__':
    main()


