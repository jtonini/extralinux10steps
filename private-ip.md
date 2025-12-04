# Private IP network for backup

Note that the assumption in the explanation is that `eth0` is the already connected
interface that remains with the UR network. `eth1` is the new one. Actual names will 
differ.

The private network will be `10.11.12.0/24`. Fewer zeros, fewer typos. 

NASes will be assigned addresses in the `10.11.12.1 .. 10.11.12.9` range. This does not
create a sub-net; it is just for organization. The workstations will start at `10.11.12.100`.
The benefit is that you can look at the IP address and know what it is attached to.

## Identify the interfaces

```
ip link
```

## Bring up the new interface for testing

```
ip link set eth1 up
ip line show eth1
```

The state should be `UP`

## Give the new interface an IP address

`ip addr add 10.11.12.100/24 dev eth1`
