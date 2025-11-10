# Linux 10.x -- additional installation steps for workstations

**NB**: _This information is current as of 10 November 2025. If you are updating this
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

## Extra libraries

While these are _generally_ needed in the workstation environment, the locations and
sources of many libraries have changed since 8.10. Additionally, some of the system objects (.so files)
that were current in 8.10 are now in compatibility libraries.

Many of the installations below could be combined into a single `dnf install ...` command,
but they are listed individually for clarity.

```
dnf install epel-release
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo
dnf install environment-modules
dnf install libcrypt\*
dnf install libgfortran\*
dnf install libGLU\*
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




