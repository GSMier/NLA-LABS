# GMRES
## To compile 
```
cd 03-lis/
```
```
module load gcc-glibc
```
```
module load lis
```

```
mpicxx -DUSE_MPI compare_gmres.cpp -I"$mkEigenInc" -I"$mkLisInc" -L"$mkLisLib" -llis -o compare_gmres
```

## Run
```
./compare_gmres
```