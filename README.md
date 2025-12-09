# Linux 10.x -- additional installation steps for workstations

**NB**: _This information is current as of 9 December 2025. If you are updating this
document, please change this date._

## Disable SELinux

SELinux has its purposes, but it interferes with several scientific packages. The workstations
do not run webservers, and they are only accessible through the UR VPN or on-campus. Consequently, 
the benefits of disabling SELinux far outweigh the perceived benefits of having it 
remain active.

First, disable it within the current session. The effect will be immediate, and you thus
do not need to reboot to observe the effect. It is important that disabling SELinux be done
before other installs. If it is not, then you will see problems with mysteriously changing
permissions and owners, most obviously when you install components under `sssd`, below.
```bash
setenforce 0
```

Second, edit the `/etc/selinux/config` file so that the first non-comment line reads:
```
SELINUX=disabled
```
On reboot, SELinux will not return to life.

## Update the installed image

The install media will be behind the current release even if you created the install media
moments ago.

`dnf update`

## Extra libraries

While these are _generally_ needed in the workstation environment, the locations and
sources of many libraries have changed since 8.10. Additionally, some of the system objects (.so files)
that were current in 8.10 are now in compatibility libraries.

### Truly required for normal operation

Many of the installations below could be combined into a single `dnf install ...` command,
but they are listed individually for clarity.

```bash
dnf install epel-release
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo
dnf install environment-modules
dnf install libcrypt\*
dnf install libgfortran\*
dnf install libGLU\*
dnf install libXmu\*
dnf install libffi\*
```

### Possibly required for builds

The following may not be required unless you need to build a package locally. It takes only a little time
to install them, and they do not take up much space. Many of the development tools provide one or more
system objects that will be required by pacakages built with the tool, implying that installing the
development tools will save you time down the road.

Once installed, the usual `dnf update` process will update them if required. Also note that several of
the packages (perhaps most) may already be present depending on which install image was chosen when
Linux 10 was installed.

```bash
dnf install bc 
dnf install bison\*
dnf install bzip2 bzip2-devel
dnf install cmake\* 
dnf install g++ 
dnf install gcc gcc-gfortran gcc-c++ 
dnf install glibc-headers 
dnf install kernel-headers 
dnf install libnsl 
dnf install ncurses\*
dnf install libXt-devel libX11-devel libXext-devel 
dnf install make 
dnf install mesa\* 
dnf install mesa-lib\* 
dnf install netcdf-devel openmpi-devel fftw-devel 
dnf install openmpi\* 
dnf install patch 
dnf install perl 
dnf install python3-devel 
dnf install tcsh 
dnf install tcl-devel 
dnf install swig\* 
dnf install util-linux 
dnf install wget 
dnf install zlib\* 
```

### Extra legacy libraries.

For all of these legacy libraries, you will need to create the appropriate symbolic links. For example,
if you need `libffi.so.6`, the most recent version you are likely to find is `libffi.so.6.0.2`. Consequently,
you will need these links in `/usr/lib64`:

```bash
ln -s libffi.so.6.0.2 libffi.so.6
ln -s libffi.so.6.0.2 libffi.so.6.0
```

Libraries:
```bash
libffi.so.6
```


## NVIDIA drivers and CUDA 

Locating the rpm requires looking for it on the nvidia.com site. Search for "rocky linux 10 driver rpm download site:nvidia.com" and you will likely find yourself on the https://developer.nvidia.com/datacenter-driver-downloads page, which does have a Rocky 10 section after following the prompts. The one most recently retrieved is this one:
```
wget https://developer.download.nvidia.com/compute/nvidia-driver/580.105.08/local_installers/nvidia-driver-local-repo-rhel10-580.105.08-1.0-1.x86_64.rpm
```

The Linux 10 drivers support CUDA versions <= 13.0. You should be able to install the current CUDA toolkit with:
```
dnf install cuda-toolkit
```

## Centralized authentication

University of Richmond uses an AD/LDAP central system to support password authentication single sign-on across most
university owned computers on campus. If the university has switched to key-based authentication before you are
reading this article, then you may skip this section entirely. If keys are used, then no other authentication methods
are attempted.

### Get the files

Two files are involved: `/etc/krb5.conf` and `/etc/sssd/sssd.conf`. These files can be obtained from other
workstations as they provide information about the AD/LDAP authentication system's servers rather than 
providing information about the workstation. At this time, LDAP requests that originate from the VPN or the
on-campus networks are assumed to be legitimate.

The kerberos file, `/etc/krb5.conf` can be dropped into its location overwriting the skeleton file that
is provided as a part of a clean installation of Linux 10. This file is world readable, so no changes to its
permissions are required.

The sssd file, `/etc/sssd/sssd.conf` will be a new file.

### Set the permissions and owners

The permissions and owners should be set appropriately:

```bash
[~]: tree -pug /etc/sssd
[drwxr-x--- root     sssd    ]  /etc/sssd
├── [drwxr-x--x root     sssd    ]  conf.d
├── [drwxr-x--x root     sssd    ]  pki
└── [-rw-r----- root     sssd    ]  sssd.conf
```

The commands needed to accomplish the above setup are:

```bash
chown -R root:sssd /etc/sssd
chmod 750 /etc/sssd
chmod 751 /etc/sssd/conf.d
chmod 751 /etc/sssd/pki
chmod 640 /etc/sssd/sssd.conf
```

### Location of the certificates

If you obtained the `sssd.conf` file from a Linux 8 or 9 computer, you must correct the location of the
certificates. The following shows the old location followed by the new, correct location:

```bash
< 	ldap_tls_cacert = /etc/openldap/cacerts/ca-chain.pem
---
> 	ldap_tls_cacert = /etc/pki/tls/certs/ca-bundle.crt
```

### Update trust model and crypographic algorithms

These commands make the correct settings for the University of Richmond environment.

```bash
update-crypto-policies --set LEGACY
update-ca-trust
```

### Start the authentication system

You should now be able to start the authentication system:

```bash
systemctl enable sssd
systemctl start sssd
```

You can check its operation with 

```bash
systemctl --no-pager status sssd
```

The output should look something like this:

```
● sssd.service - System Security Services Daemon
     Loaded: loaded (/usr/lib/systemd/system/sssd.service; enabled; preset: enabled)
     Active: active (running) since Thu 2025-11-06 16:03:38 EST; 3 days ago
 Invocation: 0f98932c481d427aa650a531981d4305
   Main PID: 307461 (sssd)
      Tasks: 5 (limit: 820631)
     Memory: 133.3M (peak: 133.9M)
        CPU: 24.060s
     CGroup: /system.slice/sssd.service
             ├─307461 /usr/sbin/sssd -i --logger=files
             ├─307462 /usr/libexec/sssd/sssd_be --domain default --logger=files
             ├─307463 /usr/libexec/sssd/sssd_nss --logger=files
             ├─307464 /usr/libexec/sssd/sssd_pam --logger=files
             └─307465 /usr/libexec/sssd/sssd_autofs --logger=files
```

### Setting up the /usr/local software from the NAS

Add this entry to `/etc/fstab`:

`141.166.186.35:/mnt/usrlocal/8  /usr/local/chem.sw  nfs     ro,nosuid,nofail,_netdev,bg,timeo=10,retrans=2 0 0`

```bash
mkdir -p /usr/local/columbus/Col7.2.2_2023-09-06_linux64.ifc_bin
cd /usr/local/columbus/Col7.2.2_2023-09-06_linux64.ifc_bin
ln -s /usr/local/chem.sw/Columbus Columbus
```

```bash
cd /usr/local
for f in $(ls -1 chem.sw); do ln -s "chem.sw/$f" "$f"; done
```

## NVIDIA

**Achtung!** *This process is prone to minor mistakes that cause installation failures.*

NOTE: It is beneficial to have the display driver installed even on the workstations where the 
GPU is poorly suited for calculations. The video response is generally better when the display
is driven from the GPU card / driver combo.

### Preliminary work


The driver and CUDA are both *built* on the workstation. For the build to take place, the headers and development
system for the present kernel must be present.

```bash
dnf config-manager --set-enabled crb
dnf install kernel-headers\*
dnf install kernel-devel\*
```

### Driver installation

#### Get the driver

Go to https://www.nvidia.com/en-us/drivers/ and choose the correct driver.

1. Enter the model of the GPU.
2. Choose Rocky Linux 10 as the OS.
3. Click "Find"
4. Click "View"
5. Click "Download"

#### Install the driver

The driver must be installed before the remainder of the steps. CUDA and the GDS rely on the driver's
presence. 

blah blah blah

```bash
systemctl stop gdm
systemctl disable gdm
shutdown -r now
```

Proceed with the driver installation.

```bash
systemctl enable gdm
systemctl start gdm
```

### Cuda installation

Note: Cuda 11.7 is resident on the NAS that supplies `/usr/local` to the workstations. Choice of CUDA 
versions is done through the `alternatives` system.

As with the NVIDIA drivers, CUDA is obtained via a webpage: https://developer.nvidia.com/cuda-downloads

1. Click `Linux`
2. Click `x86_64`
3. Click `Rocky`
4. Click `10`
5. Click `rpm (local)`

You will get a file with a name like this: `cuda-repo-rhel10-13-0-local-13.0.1_580.82.07-1.x86_64.rpm`
(It will probably say 13.1.xxx because that's the current version of Cuda.) The advantage of using the 
local rpm install is that the rpm itself can be copied to another workstation, and the installation 
repeated. *Note: if you only had one workstation, you might prefer the network rpm.*

The installation of the CUDA toolkit involves two parts: [1] Installing the repo rpm, and [2] installing the
contents of the repo. The first step only makes the repo available for additional actions. These two steps will 
install the toolkit.

```bash
dnf install cuda-repo-rhel10-13*.rpm
dnf install cuda-toolkit
```

The installation will set the links correctly, and you can check the installation with

`/usr/local/cuda/bin/nvcc --version`

And you should see something like this (depending on the exact version you installed):

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```


### GPU Direct Storage Installation

The GPU Direct Storage software, a.k.a., GDS (GPU Direct Storage) is present in the CUDA
toolkit rpm. It provides higher I/O speed when used with NVMe M.2 drives that are in the CPU-direct
slot on the motherboard. Installation is simple:

```bash
dnf install nvidia-gds
```

The package is quite small. To see the default configuration (and to validate the GDS is aware of your setup),
issue this command:

`/usr/local/cuda/gds/tools/gdscheck -p`

Each of these parameters is set/unset in the file `/etc/cufile.json` It usually is not necessary to "tune" this
file, but in cases where the GPU support for GDS is marginal, it may be required. Here are the boundaries:

1. GTX 1080 and earlier cannot support GDS.
2. 3090 -- works well with GDS.
3. 4090 -- only works with the Enterprise driver (the one you download from Nvidia, rather than the one in some repos).

### GPU Direct Storage Assessment

Save this file as `gdsread.cu`, or something similar.

```c
#include <fcntl.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufile.h>

int main(int argc, char **argv) {
    const char *path = argv[1]; size_t sz = strtoull(argv[2], NULL, 10);
    int fd = open(path, O_RDONLY | O_DIRECT);
    cuFileDriverOpen();
    CUfileDescr_t d = {}; d.handle.fd = fd; d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t h; cuFileHandleRegister(&h, &d);
    void *dptr; cudaMalloc(&dptr, sz);
    ssize_t n = cuFileRead(h, dptr, sz, 0, 0); cudaDeviceSynchronize();
    cuFileHandleDeregister(h); cuFileDriverClose(); close(fd); cudaFree(dptr);
    return (n < 0);
}
```

Compile it: `nvcc -O2 -o gdsread gdsread.cu -lcufile`

Find a big file on the NVMe drive, or create one. Then, 

```bash
FILE=/mnt/nvme0/bigfile
SIZE=$((32 * 1024 * 1024 * 1024))  # 32G .. at least larger than bigfile.

/usr/bin/time -v "GDS read: %E real, %S sys, %U user" ./gdsread "$FILE" "$SIZE"
```

Now compare to the default method to read this file:

`/usr/bin/time -v "CPU read: %E real, %S sys, %U user" dd if="$FILE" of=/dev/null bs=4M status=none`

































































