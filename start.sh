#!/bin/bash

help_message="Usage: ./start.sh [options]
  Options:
    -k    --kernel        Directory where kernel located. ./kernel/ by default.
    -m    --image         Directory where image located (one chanel per file). ./image/ by default
    -c    --compile       Compile before executing
    -co   --compileopt    Compile with optimization
    -h    --help          Show help message"
kernel="kernel/"
image="image/"
while true; do
  case $1 i
    -c|--compile)
      comp=true;
      shift
      ;;
    -co|--compileopt)
      comp=true;
      flags="-DCMAKE_BUILD_TYPE=Release";
      shift
      ;;
    -k|--kernel)
      kernel=$2
      shift 2
    ;;
    -im|--image)
      image=$2
      shift 2
      ;;
    -h|--help)
      echo "$help_message"
      exit 0;
    ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1 ;;
    :)
      echo "Option -$OPTARG requires an numerical argument." >&2
      exit 1 ;;
    *)
      break
      ;;
  esac
done

mkdir -p ./cmake-build-debug;
pushd ./cmake-build-debug  > /dev/null || exit 1

if [[ "$comp" = true ]]; then
  echo Compiling...
  cmake "$flags" -G"Unix Makefiles" ..;
  make;
fi;



popd > /dev/null || exit 1
./cmake-build-debug/CNN_convolution_2D "$image" "$kernel"
