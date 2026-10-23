from dataclasses import fields
from pathlib import Path
from typing import Any

from healthcheck.sysinfo import nic, topology
from healthcheck.sysinfo.gpu import Gpu
from healthcheck.sysinfo.nic import Hca, LinkLayer
from healthcheck.sysinfo.topology import NumaNode, Placement, numa_layout
from healthcheck.util.device import BusId, PciDevice


def pci_device(root: Path, path: str, numa_node: int) -> BusId:
    """A PCI device at /sys/devices/<path>, linked from /sys/bus/pci/devices like the real one."""
    device = root / "devices" / path
    device.mkdir(parents=True)
    (device / "numa_node").write_text(f"{numa_node}\n")
    (device / "vendor").write_text("0x15b3\n")
    (device / "device").write_text("0x1021\n")
    links = root / "bus" / "pci" / "devices"
    links.mkdir(parents=True, exist_ok=True)
    (links / device.name).symlink_to(device)
    return BusId.parse(device.name)


def make[T](cls: type[T], **values: Any) -> T:
    """cls with every field None but values."""
    return cls(**{field.name: None for field in fields(cls)} | values)


def test_from_sysfs_root_port(tmp_path):
    """Fir: GPU0 and mlx5_4 below one root port, mlx5_5 below the next one."""
    gpu0 = pci_device(tmp_path, "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:06:00.0", 3)
    mlx5_4 = pci_device(tmp_path, "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:01.0/0000:03:00.0", 3)
    mlx5_5 = pci_device(tmp_path, "pci0000:00/0000:00:03.1/0000:07:00.0/0000:08:01.0/0000:09:00.0", 3)
    gpu, hca, other = (PciDevice.from_sysfs(address, tmp_path, name=None) for address in (gpu0, mlx5_4, mlx5_5))
    assert gpu == PciDevice(
        name=None, numa_node=3, address=gpu0, vendor="0x15b3", device="0x1021", root_port=BusId(0, 0, 1, 1)
    )
    assert hca.same_root_port(gpu)
    assert not other.same_root_port(gpu)


def test_from_sysfs_on_host_bridge_or_missing(tmp_path):
    integrated = PciDevice.from_sysfs(pci_device(tmp_path, "pci0000:00/0000:00:14.0", -1), tmp_path, name=None)
    assert (integrated.numa_node, integrated.root_port) == (None, None)
    assert not integrated.same_root_port(integrated)
    missing = PciDevice.from_sysfs(BusId(0, 0x99, 0, 0), tmp_path, name=None)
    assert (missing.numa_node, missing.vendor, missing.root_port) == (None, None, None)
    assert PciDevice.from_sysfs(None, tmp_path, name="rxe0").address is None


def test_to_json_nests_pci():
    hca = make(Hca, name="mlx5_2", numa_node=2, address=BusId(0, 0x43, 0, 0), root_port=BusId(0, 0x40, 1, 1), ports=[])
    assert hca.to_json() == {
        "name": "mlx5_2",
        "hca_type": None,
        "fw_ver": None,
        "board_id": None,
        "netdevs": None,
        "ports": [],
        "pci": {"address": "0000:43:00.0", "vendor": None, "device": None, "numa_node": 2, "root_port": "0000:40:01.1"},
    }
    assert make(Gpu, index=0).to_json()["index"] == 0


def test_hcas(tmp_path):
    address = pci_device(tmp_path, "pci0000:40/0000:40:01.1/0000:41:00.0/0000:42:00.0/0000:43:00.0", 2)
    dev = tmp_path / "class" / "infiniband" / "mlx5_2"
    (dev / "ports" / "1").mkdir(parents=True)
    (dev / "device").symlink_to(tmp_path / "bus" / "pci" / "devices" / str(address))
    (dev / "fw_ver").write_text("28.48.1000\n")
    for name, value in [("state", "4: ACTIVE"), ("rate", "200 Gb/sec (2X NDR)"), ("link_layer", "InfiniBand")]:
        (dev / "ports" / "1" / name).write_text(f"{value}\n")
    [hca] = nic.hcas(tmp_path)
    assert (hca.name, hca.fw_ver, hca.address, hca.numa_node) == ("mlx5_2", "28.48.1000", address, 2)
    assert hca.root_port == BusId(0, 0x40, 1, 1)
    assert hca.ports == [
        {
            "port": 1,
            "state": "ACTIVE",
            "phys_state": "",
            "rate": "200 Gb/sec (2X NDR)",
            "rate_gbps": 200.0,
            "link_layer": "InfiniBand",
        }
    ]
    assert hca.link_layers == [LinkLayer.INFINIBAND]


def test_link_layers():
    vpi = make(Hca, ports=[{"link_layer": "Ethernet"}, {"link_layer": "InfiniBand"}])
    assert vpi.link_layers == [LinkLayer.ETHERNET, LinkLayer.INFINIBAND]
    assert make(Hca, ports=[{"link_layer": "Unknown"}]).link_layers == []
    assert make(Hca, ports=[]).link_layers == []


def two_sockets(root: Path) -> Path:
    """
    Two sockets with hyperthreads, each split into two NUMA nodes: cpus 0-1 and 4-5 on socket 0, 2-3 and 6-7 on
    socket 1, node 2 * socket + cpu % 2. Plus an offline cpu8 and a memory-only (CXL) node4. Returns the cpuinfo.
    """
    cpuinfo = []
    for index in range(8):
        socket = index // 2 % 2
        path = root / "devices" / "system" / "cpu" / f"cpu{index}"
        (path / "topology").mkdir(parents=True)
        (path / "topology" / "physical_package_id").write_text(f"{socket}\n")
        (path / f"node{2 * socket + index % 2}").mkdir()
        cpuinfo.append(f"processor\t: {index}\nphysical id\t: {socket}\nmodel name\t: EPYC {socket}\n")
    (root / "devices" / "system" / "cpu" / "cpu8").mkdir()
    for index in range(5):
        node = root / "devices" / "system" / "node" / f"node{index}"
        node.mkdir(parents=True)
        (node / "meminfo").write_text(f"Node {index} MemTotal:       296955060 kB\nNode {index} MemFree: 1 kB\n")
    (root / "cpuinfo").write_text("\n".join(cpuinfo))
    return root / "cpuinfo"


def test_placements(tmp_path):
    two_sockets(tmp_path)
    assert topology.placements(tmp_path)[:3] == [Placement(0, 0, 0), Placement(1, 0, 1), Placement(2, 1, 2)]
    assert len(topology.placements(tmp_path)) == 8  # not the offline cpu8


def test_sockets(tmp_path):
    cpuinfo = two_sockets(tmp_path)
    sockets = topology.sockets(topology.placements(tmp_path), cpuinfo)
    assert [socket.to_json() for socket in sockets] == [
        {"name": "EPYC 0", "numa_node": None, "socket": 0, "cpus": "0-1,4-5", "cores": 4, "numa_nodes": [0, 1]},
        {"name": "EPYC 1", "numa_node": None, "socket": 1, "cpus": "2-3,6-7", "cores": 4, "numa_nodes": [2, 3]},
    ]


def test_sockets_without_physical_id(tmp_path):
    """Some VMs and containers list no physical id: every socket takes the first model name."""
    cpuinfo = two_sockets(tmp_path)
    cpuinfo.write_text("processor\t: 0\nmodel name\t: EPYC\n\nprocessor\t: 1\nmodel name\t: EPYC\n")
    assert [socket.name for socket in topology.sockets(topology.placements(tmp_path), cpuinfo)] == ["EPYC", "EPYC"]


def test_numa_nodes(tmp_path):
    two_sockets(tmp_path)
    nodes = topology.numa_nodes(topology.placements(tmp_path), tmp_path)
    assert [(node.numa_node, node.socket, node.cpus) for node in nodes] == [
        (0, 0, "0,4"),
        (1, 0, "1,5"),
        (2, 1, "2,6"),
        (3, 1, "3,7"),
        (4, None, None),
    ]
    assert nodes[1].to_json() == {"name": "node1", "numa_node": 1, "socket": 0, "cpus": "1,5", "mem_gib": 283.2}


def test_numa_layout_fir():
    """Fir: GPUs 0-1 on NUMA node 3 and 2-3 on node 2, each with an IB HCA on its PCIe switch; Ethernet on node 0."""
    roots = [BusId(0, 0, 1, 1), BusId(0, 0, 3, 1), BusId(0, 0x40, 1, 1), BusId(0, 0x40, 3, 1)]
    nodes = [make(NumaNode, numa_node=node, socket=0, cpus=f"{12 * node}-{12 * node + 11}") for node in range(4)]
    gpus = [
        make(Gpu, index=index, address=BusId(0, bus, 0, 0), numa_node=3 - index // 2, root_port=roots[index])
        for index, bus in enumerate([0x06, 0x0C, 0x46, 0x4C])
    ]
    ib = [{"link_layer": "InfiniBand"}]

    def hca(name: str, bus: int, numa_node: int, root_port: BusId | None, ports: list) -> Hca:
        return make(Hca, name=name, address=BusId(0, bus, 0, 0), numa_node=numa_node, root_port=root_port, ports=ports)

    hcas = [
        hca("mlx5_0", 0x81, 0, None, [{"link_layer": "Ethernet"}]),
        hca("mlx5_2", 0x43, 2, roots[2], ib),
        hca("mlx5_3", 0x49, 2, roots[3], ib),
        hca("mlx5_4", 0x03, 3, roots[0], ib),
        hca("mlx5_5", 0x09, 3, roots[1], ib),
    ]
    assert numa_layout(nodes, gpus, hcas, "InfiniBand") == [
        {"socket": 0, "numa": 0, "cpus": "0-11", "gpus": "", "hcas": "", "other": "mlx5_0 (81:00.0, Ethernet)"},
        {"socket": 0, "numa": 1, "cpus": "12-23", "gpus": "", "hcas": "", "other": ""},
        {
            "socket": 0,
            "numa": 2,
            "cpus": "24-35",
            "gpus": "GPU2 (46:00.0); GPU3 (4c:00.0)",
            "hcas": "mlx5_2 (43:00.0) same PCIe switch as GPU2; mlx5_3 (49:00.0) same PCIe switch as GPU3",
            "other": "",
        },
        {
            "socket": 0,
            "numa": 3,
            "cpus": "36-47",
            "gpus": "GPU0 (06:00.0); GPU1 (0c:00.0)",
            "hcas": "mlx5_4 (03:00.0) same PCIe switch as GPU0; mlx5_5 (09:00.0) same PCIe switch as GPU1",
            "other": "",
        },
    ]


def test_numa_layout_device_without_numa_affinity():
    nodes = [make(NumaNode, numa_node=0, socket=0, cpus="0-11")]
    gpus = [make(Gpu, index=0, address=BusId(0, 1, 0, 0), numa_node=None)]
    assert numa_layout(nodes, gpus, [], "InfiniBand") == [
        {"socket": 0, "numa": 0, "cpus": "0-11", "gpus": "", "hcas": "", "other": ""},
        {"socket": None, "numa": None, "cpus": None, "gpus": "GPU0 (01:00.0)", "hcas": "", "other": ""},
    ]
