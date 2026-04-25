docker run --rm -t \
  -v "/home/tomcao/Research_KyxParser/GuidedSampler:/data" \
  delcypher/jfs_build:fse_2019 \
  /home/user/jfs/build/bin/jfs \
  -branch-encoding=try-all-imncsf \
  -sanitizer-coverage=trace-pc-guard,trace-cmp \
  -keep-output-dir \
  -output-dir=/data/jfs_diverse_build \
  /data/simple_circle_fp.smt2




for i in {1..10}; do   echo "Running sampler $i...";   docker run --rm -t     -v "/home/tomcao/Research_KyxParser/GuidedSampler:/data"     delcypher/jfs_build:fse_2019     /data/jfs_diverse_build/fuzzer     -artifact_prefix=/data/solutions/sol_${i}_     -max_total_time=60     -max_len=16     -min_len=16     -seed=$((RANDOM))     -use_value_profile=1     -mutate_depth=20     /data/jfs_diverse_build/corpus > /dev/null 2>&1;      if ls /home/tomcao/Research_KyxParser/GuidedSampler/solutions/sol_${i}_* 1> /dev/null 2>&1; then     echo "Successfully found sample $i";   else     echo "Sample $i failed (timeout)";   fi; done