for i in {29500..30000}
do
   fuser -k $i/tcp
done

nvidia-smi  | awk '/ C / { print $5 }' | xargs -I {} kill -9 {}
