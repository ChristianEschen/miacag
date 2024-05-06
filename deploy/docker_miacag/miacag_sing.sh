sudo docker build --progress=plain -t miacag_sing .

sudo docker save -o singularity_miacag.tar miacag_sing
​
sudo singularity build --sandbox s_container docker-archive://singularity_miacag.tar
​
sudo singularity build miacag_singularity.sif s_container
​
SINGULARITY_TMPDIR=$PWD/singularity/tmp SINGULARITY_CACHEDIR=$PWD/singularity/cache singularity shell --nv -B $PWD:$PWD /home/alatar/miacag/singularity_miacag/miacag_singularity.sif


SINGULARITY_TMPDIR=$PWD/singularity/tmp SINGULARITY_CACHEDIR=$PWD/singularity/cache singularity shell --nv -B $PWD:$PWD /home/alatar/miacag/singularity_miacag/file_out_curl


file_out_curl