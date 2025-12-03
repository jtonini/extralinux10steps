# Linux 10.x -- additional installation steps for workstations

**NB**: _This information is current as of 11 November 2025. If you are updating this
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
```
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

```
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

```
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

```
ln -s libffi.so.6.0.2 libffi.so.6
ln -s libffi.so.6.0.2 libffi.so.6.0
```

Libraries:
```
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

```
[~]: tree -pug /etc/sssd
[drwxr-x--- root     sssd    ]  /etc/sssd
├── [drwxr-x--x root     sssd    ]  conf.d
├── [drwxr-x--x root     sssd    ]  pki
└── [-rw-r----- root     sssd    ]  sssd.conf
```

The commands needed to accomplish the above setup are:

```
chown -R root:sssd /etc/sssd
chmod 750 /etc/sssd
chmod 751 /etc/sssd/conf.d
chmod 751 /etc/sssd/pki
chmod 640 /etc/sssd/sssd.conf
```

### Location of the certificates

If you obtained the `sssd.conf` file from a Linux 8 or 9 computer, you must correct the location of the
certificates. The following shows the old location followed by the new, correct location:

```
< 	ldap_tls_cacert = /etc/openldap/cacerts/ca-chain.pem
---
> 	ldap_tls_cacert = /etc/pki/tls/certs/ca-bundle.crt
```

### Update trust model and crypographic algorithms

These commands make the correct settings for the University of Richmond environment.

```
update-crypto-policies --set LEGACY
update-ca-trust
```

### Start the authentication system

You should now be able to start the authentication system:

```
systemctl enable sssd
systemctl start sssd
```

You can check its operation with 

```
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

```
mkdir -p /usr/local/columbus/Col7.2.2_2023-09-06_linux64.ifc_bin
cd /usr/local/columbus/Col7.2.2_2023-09-06_linux64.ifc_bin
ln -s /usr/local/chem.sw/Columbus Columbus
```

```
cd /usr/local
for f in $(ls -1 chem.sw); do ln -s "chem.sw/$f" "$f"; done
```
