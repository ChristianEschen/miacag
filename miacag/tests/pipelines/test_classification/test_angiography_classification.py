import subprocess
import os


def test_submit_pipeline_cpu():
    os.system(
        """python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=192.168.8.116 \
        --master_port=1237 \
        scripts/angiography_classifier/submit_pipeline.py \
        --cpu \
        True""")



if __name__ == '__main__':
    test_submit_pipeline_cpu()
