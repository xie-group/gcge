# GCGE文件结构

app

config

src

test

# GCGE配置编译

GCGE可以无需任何外部包独立编译运行，它内置的矩阵结构是
行/列压缩存储的矩阵 和 稠密矩阵。可以完成OpenMP加速。

## 下载安装GCGE包

下载地址为https://github.com/Matrials-Of-Numerical-Algebra/GCGE

### Windows

首先安装Dev-C++
https://sourceforge.net/projects/orwelldevcpp/

利用Dev-C++打开项目文件 test/TestOPS.dev

### Matlab

Matlab R2017a 及以上的版本可以使用, 可以直接使用 test/app_matlab.mexw64.

如果需要在matlab下编译源码生成 mexw64
首先在官网
https://ww2.mathworks.cn/support/requirements/supported-compilers.html
查看混合编译的支持和兼容的编译器.

如果是windows环境请下载MinGW
其它系统参照对应的编译环境

启动matlab 并运行 test/makefile_matlab.m
生成 mexw64

接口函数是
gcge_matlab.c

### Linux

若Linux系统中已安装git命令，执行：
`git clone https://github.com/Matrials-Of-Numerical-Algebra/GCGE`

可以使用git pull命令来获得当前包的更新内容：

`cd GCGE`

`git pull`

在文件加/GCGE/config中有适用于不同环境的Makefile文件，选择你需要的文件并修改

`/GCGE/test/Makefile:#include ../config/make.MPI.inc`

并将CFLAG的-qopenmp改为-fopenmp（编译环境决定）。

用GCGE进行特征值求解时，在不同接口测试函数test_app_*文件中调用

`TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);`

当然我们也可以进行一些其他的计算，如对（多）向量进行正交化TestOrth(matB,ops)，对应的测试函数为test_orth.c；矩阵多向量乘TestMultiVec(matA,ops)，对应的测试函数为test_multi_vec.c。即在相应的接口测试函数中调用相关函数即可。

如果要测试不同接口，需要修改文件/test/main.c，同时我们还要相应地修改文件夹/src中的ops_config.h文件中变量宏定义。


# 外部包编译

GCGE支持的外部包有MPICH、BLAS、LAPACK、HYPRE、PETSc、SLEPc和PHG，其中MPICH、BLAS和LAPACK都可以在配置PETSc的时候进行下载和安装，当然我们可以单独安装这些包。下面的安装步骤是基于Linux系统介绍的，在安装之前请确保Linux系统已安装gcc、g++、gfortran、make和python等。

以gcc为例，在终端执行：

`gcc -v`

来判断是否已经安装gcc编译器并查看其版本，执行：

`sudo apt install gcc`

来安装gcc。

## MPICH

最简单的一种方式是通过执行：

`sudo apt install mpich`

来下载和安装MPICH，这时会安装在默认目录下，执行：

`which mpicc` 或`which mpicc`

可以定位安装路径。

也可以先下载MPICH的安装包，下载地址为https://www.mpich.org/downloads/，将压缩包放在指定文件夹（如/home/wzj/package)，在该路径下执行：

`tar xzvf mpich-3.4.tar.gz`

来解压软件包，再进入该包的目录下进行编译，执行：

`cd mpich-3.4`

`./configure --with-device=ch3` 

当然在配置的时候我们可以指定MPICH的安装路径，在这里建议安装在默认路径下。

执行：

`make`

`make install`

进行编译和安装，同样可以检测和定位安装路径。安装完成后可以用/mpich-3.4/examples中的例子程序测试是否安装成功，如测试cpi.c，执行：

`make cpi`

`mpirun -np 2 ./cpi`

在用第一种方式安装时我们也可以通过包里的例子程序进行测试，如测试cpi.c，执行：

`mpicc cpi.c -o cpi`

`mpirun -np 2 ./cpi`

我们可以通过命令：

`which mpicc`

来判断mpicc的位置

## HYPRE

下载HYPRE安装包，下载地址为https://launchpad.net/ubuntu/+source/hypre/2.18.2-1，将压缩包放在指定文件夹，并解压，在该路径下执行：

`tar xzvf hypre-2.18.2.tar.gz`

`cd hypre-2.18.2`

进入该包的文件夹src目录下进行配置和安装执行：

`cd src`

`./configure`

`make install`

安装完成后可以用/src/example中的例子程序测试是否安装成功，如测试ex1.c，执行：

`make ex1`

`mpirun -np 2 ./ex1`

在.bashrc文件中写入

`export HYPRE_DIR="/home/wzj/package/hypre-2.18.2/src/hypre`

## PETSc

SLEPC是基于PETSc的，因此我们首先安装PETSc，在配置时安装BLAS和LAPACK。

下载PETSc安装包，下载地址为https://www.mcs.anl.gov/petsc/download/index.html，将压缩包放在指定文件夹（如/home/wzj/package)并解压，在该路径下执行：

`wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.14.3.tar.gz`

`tar xzvf petsc-3.14.3.tar.gz`

`cd petsc-3.14.3`

PETSC主页对其配置进行了详细说明，见https://www.mcs.anl.gov/petsc/documentation/installation.html。首先需要配置两个环境变量PETSC_ARCH和PETSC_DIR，在~/.bashrc文件中写入指定内容，在当前用户目录下执行：

`vim ~/.bashrc`

写入

`export PETSC_DIR="/home/wzj/package/petsc-3.14.3"`

`export PETSC_ARCH=linux-gnu-c-debug`(linux-gnu-c-debug是随便取的一个路径名)

执行：

`source ~/.bashrc`

使该文件生效。

如果将MPICH安装到默认路径则不需要在配置时指定MPICH的安装路径，在配置时需要同时下载和安装BLAS和LAPACK，执行：

`./configure --download-fblaslapack`

配置完成后会出现make的具体内容，安装提示信息进行编译即可，如执行：

`make PETSC_DIR=/home/wzj/package/petsc-3.14.3 PETSC_ARCH=linux-gnu-c-debug all`

`make PETSC_DIR=/home/wzj/package/petsc-3.14.3 PETSC_ARCH=linux-gnu-c-debug check`

完成后，可以用src中的例子程序进行测试，如测试/src/ksp/ksp/tutorials中的ex1.c，执行：

`make ex1`

`./ex1`

## SLEPc

下载SLEPc安装包，下载地址为https://slepc.upv.es/download/，将压缩包放在指定文件夹（如/home/wzj/package)并解压，在该路径下执行：

`tar xzvf slepc-3.14.1.tar.gz`

`cd slepc-3.14.1`

配置环境变量SLEPC_DIR，在当前用户目录下执行

`vim ~/.bashrc`

写入

`export SLEPC_DIR="/home/wzj/package/slepc-3.14.1"`

执行：

`source ~/.bashrc`

使该文件生效。

配置后会出现make的具体内容，按照提示信息进行编译，执行：

`./configure`

`make SLEPC_DIR=/home/wzj/package/slepc-3.14.1 PETSC_DIR=/home/wzj/package/petsc-3.14.3 PETSC_ARCH=linux-gnu-c-debug`

`make SLEPC_DIR=/home/wzj/package/slepc-3.14.1 PETSC_DIR=/home/wzj/package/petsc-3.14.3 check`

完成后，可以用src中的例子程序进行测试，如测试/src/eps/tutorials中的ex1.c，执行：

`make ex1`

`./ex1`

## PHG

下载PHG安装包，下载地址为http://lsec.cc.ac.cn/phg/download.htm，将压缩包放在指定文件夹（如/home/wzj/package)并解压，在该路径下执行：

`tar xjvf phg-0.9.5-20200727.tar.bz2`

`cd phg-0.9.5/`

由于编译后生成的src中的refine.o和coarsen.o文件可能并不支持并行计算，根据MPI的版本将相应的obj文件夹下的refine.o和coarsen.o文件复制到src下，并删除src文件夹中的refine.c和coarsen.c，即执行：

`cp obj/mpich-x86_64/double_int/*.o src/.`

`cd src`

`rm coarsen.c refine.c`

配置环境变量PHG_DIR，执行：

`vim ~/.bashrc`

写入

`export PHG_DIR="/home/wzj/package/phg-0.9.5"`

执行：

`source ~/.bashrc`

使该文件生效，再进行配置和编译，执行：

`./configure --enable-shared`

`make`

完成后，可以用examples文件夹中的例子程序进行测试，如测试simplest.c，执行：

`make simplest`

`mpirun -np 2 ./simplest`

由于在配置的时候使用选项--enble shared生成了一个开放的库文件libphg.so，该文件的默认位置在/phg-0.9.5/src中，我们要将该文件复制到/usr/lib下，执行：

`sudo cp /home/wzj/package/phg-0.9.5/src/libphg.so /usr/lib/.`

## GCGE 测试外部接口

下载GCGE安装包，下载地址为https://github.com/Matrials-Of-Numerical-Algebra/GCGE，如果Linux系统中已安装git命令，执行：

`git clone https://github.com/Matrials-Of-Numerical-Algebra/GCGE`

可以使用git pull命令来获得当前包的更新内容：

`cd GCGE`

`git pull`

在文件加/GCGE/config中有适用于不同环境的Makefile文件，选择你需要的文件并修改

`/GCGE/test/Makefile:#include ../config/make.MPI.inc`

并将CFLAG的-qopenmp改为-fopenmp（编译环境决定）。

用GCGE进行特征值求解时，在不同接口测试函数test_app_*文件中调用

`TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);`

当然我们也可以进行一些其他的计算，如对（多）向量进行正交化TestOrth(matB,ops)，对应的测试函数为test_orth.c；矩阵多向量乘TestMultiVec(matA,ops)，对应的测试函数为test_multi_vec.c。即在相应的接口测试函数中调用相关函数即可。

如果要测试不同接口，需要修改文件/test/main.c，同时我们还要相应地修改文件夹/src中的ops_config.h文件中变量宏定义。

（1）如我们要测试SLEPc接口：

`在/test/main.c中选择TestAppSLEPC(argc,argv)，注释掉其他接口`

`将/src/ops_config.h中OPS_USEMPI和OPS_USESLEPC的宏定义设置为1`

`将/src/ops_config.h中OPS_USEMPI和OPS_USEINTEL_MKL的宏定义设置为0`(编译环境决定)

然后进入/test进行编译，执行：

`make`

得到可执行文件TestOPS.exe，执行以下命令可得到结果：

`mpirun -np 2 ./TestOPS`

（2）如我们要测试PHG接口

`在/test/main.c中选择TestPHG(argh,argv)，并注注释掉其他接口`

`将/src/ops_config.h中OPS_USEMPI和OPS_USEPHG的宏定义设置为1`

在/test/get_mat_phg.c中，可以修改pre_refine的值来改变计算规模：

`int pre_refines = 1`

其余接口和操作的测试都是类似的。

此外参数设置在文件test_sol_eig_gcg.c中，可以在程序中直接修改参数值（要重新编译），也可以通过命令行参数修改。如要修改GCG的最大迭代次数，执行：

`./TestOPS.exe -gcge_max_niter 100`

