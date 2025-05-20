# executed by Snakemake via `script:` (same Python process)
from vaft.code import init_snakemake_logger
from vaft.database        import store_shot_as_json, init_pool
import os, sys, logging
    
logger = init_snakemake_logger(snakemake, std_opt=True) 

shot   = int(snakemake.wildcards.shot)
# Create parent directory for the output file
os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
print('save path: ', snakemake.output[0])

init_pool()
rc = store_shot_as_json(shot = shot,output_path = snakemake.output[0])
print(rc)

# Check if rc is False or a non-zero integer
if rc is True:
    logger.info("store_shot_as_json() completed successfully")
else:
    logger.error("store_shot_as_json() failed with return code %s", rc)
    raise ValueError("store_shot_as_json() failed")
