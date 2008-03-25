#! /bin/sh

set -e

F77=gfortran
SOURCE_FILES="xerbla.f"
LIBRARY=libxerbla.a

rm -f $LIBRARY

for i in $SOURCE_FILES; do
  DESTNAME=${i%.f}.o
  $F77 -fpic -c -o $DESTNAME $i
  ar -rc $LIBRARY $DESTNAME
done

ranlib $LIBRARY
