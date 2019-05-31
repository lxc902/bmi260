if [ "$#" -ne 5 ]; then
  echo "Illegal number of parameters"
  CMD="move_files.sh"
  echo "Usage:        ./$CMD [src1] [src2] [des1] [des2] [number to move]"
  echo "Sample usage: ./$CMD train/image train/label test truth 95"
  exit
fi
NUM=$5
if [ "$NUM" -le 0 ]; then
  echo "number($NUM) is too small"
fi

IN1=$1
IN2=$2
OUT1=$3
OUT2=$4
mkdir -p $OUT1
mkdir -p $OUT2
cnt=$NUM
for img in `ls $IN1`; do
  mv $IN1/$img $OUT1/$img
  mv $IN2/$img $OUT2/$img
  cnt=$((cnt-1))
  if [ "$cnt" == "0" ]; then
    break
  fi
done

