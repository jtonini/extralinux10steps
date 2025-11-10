# Linux 10.x -- additional installation steps for workstations

# Disable SELinux

SELinux has its purposes, but it interferes with several scientific packages. The workstations
do not run webservers, and they are only accessible through the UR VPN or on-campus. Consequently, 
the benefits of disabling SELinux far outweigh the perceived benefits of having it 
remain active.

First, disable it within the current session. The effect will be immediate, and you thus
do not need to reboot to observe the effect.
```
setenforce 0
```

Second, edit the `/etc/selinux/config` file so that the first non-comment line reads:
```
SELINUX=disabled
```


# Extra libraries

While these are generally needed in the workstation environment, the locations and
sources of many have changed since 8.10. Additionally, some of the system objects (.so files)
that were current in 8.10 are now in compatibility libraries.

Many of the installations below could be combined into a single `dnf install ...` command,
but they are listed individually for clarity.

```
dnf install epel-release
dnf install environment-modules
dnf install libcrypt\*
dnf install libgfortran\*
dnf install libGLU\*
```
