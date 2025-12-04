# Private IP network for backup

Note that the assumption in the explanation is that `eth0` is the already connected
interface that remains with the UR network. `eth1` is the new one. Actual names will 
differ.

The private network will be `10.11.12.0/24`. Fewer zeros, fewer typos. Also, UR does not use
the `10.0.0.0/8` address space; we use `172.16.0.0/12` for our private IPs like the wireless
networks and the computers like Spydur in the data center.

NASes will be assigned addresses in the `10.11.12.1 .. 10.11.12.9` range. This does not
create a sub-net; it is just for organization. 

The workstations will start at `10.11.12.100`.

The switch will be `10.11.12.200`. This will allow us to login to the switch to configure 
it. 

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

```
ip addr add 10.11.12.100/24 dev eth1
ip addr show eth1
```

## Provide a route

```
ip route add 10.0.0.0/8 dev eth1
ip route | grep eth1
```

You should see something like `10.0.0.0/8 dev eth1 scope link`

## Get rid of any buggy config from the install of 10.1

```
nmcli connection delete eth1 2>/dev/null
```

Now, create the new, "real" connection (you could do with with `nmtui`
interactively). Note that there is no gateway and no DNS.

```
nmcli connection add \
    type ethernet \
    ifname eth1 \
    con-name private-eth1 \
    ipv4.method manual \
    ipv4.addresses 10.11.12.100/24 \
    ipv4.gateway "" \
    ipv4.dns "" \
    autoconnect yes
```

## Create a permanent route

Starting with Linux 10.0, the route is stored with the connection.

```
nmcli connection modify private-eth1 +ipv4.routes "10.0.0.0/8"
nmcli connection show private-eth1 | grep ipv4.routes
```

You will see something like `ipv4.routes: 10.0.0.0/8`

Verify it is live like this:

```
ip route get 10.7.8.9 # Any address in 10.0.0.0/8 will work...
```

You should see `10.7.8.9 dev eth1`

```
ip route get 8.8.4.4
```

Should show `8.8.4.4 dev eth0`

## Speed up the decision making for packet routing

```
sudo nmcli connection modify private-eth1 ipv4.route-metric 200
sudo nmcli connection modify "System eth0" ipv4.route-metric 1000
```



