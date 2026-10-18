import math
import unittest

import pandas as pd

from ml.directional import aggregate_packets, flow_key


def pk(ts, length, src="A", sport=1000, dst="B", dport=80, proto="TCP"):
    return dict(timestamp=ts, length=length, src_ip=src, src_port=sport, dst_ip=dst, dst_port=dport, protocol=proto)


class TestDirectional(unittest.TestCase):
    def test_reverse_key_is_distinct(self):
        f = pk(0, 1)
        r = pk(0, 1, src="B", sport=80, dst="A", dport=1000)
        self.assertNotEqual(flow_key(f), flow_key(r))

    def test_no_bidirectional_merge_and_known_features(self):
        pkts = pd.DataFrame([pk(0, 100), pk(1, 200), pk(2, 300),
                             pk(0.5, 50, src="B", sport=80, dst="A", dport=1000)])
        flows = aggregate_packets(pkts)
        self.assertEqual(len(flows), 2)
        fwd = flows[flows.src_ip == "A"].iloc[0]
        self.assertEqual(fwd.packet_count, 3)
        self.assertEqual(fwd.byte_count, 600)
        self.assertAlmostEqual(fwd.duration, 2.0)
        self.assertAlmostEqual(fwd.mean_pkt_len, 200.0)
        self.assertAlmostEqual(fwd.std_pkt_len, math.sqrt(20000 / 3))
        self.assertAlmostEqual(fwd.mean_iat, 1.0)
        self.assertAlmostEqual(fwd.pkts_per_sec, 1.5)
        rev = flows[flows.src_ip == "B"].iloc[0]
        self.assertEqual(rev.packet_count, 1)
        self.assertTrue(math.isnan(rev.mean_iat))
        self.assertTrue(math.isnan(rev.pkts_per_sec))  # zero duration -> undefined, not invented

    def test_idle_timeout_splits_flow(self):
        pkts = pd.DataFrame([pk(0, 10), pk(10, 10), pk(100, 10)])
        flows = aggregate_packets(pkts, idle_timeout=60)
        self.assertEqual(sorted(flows.packet_count), [1, 2])
        long = aggregate_packets(pkts, idle_timeout=500)
        self.assertEqual(len(long), 1)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            aggregate_packets(pd.DataFrame([{"timestamp": 0, "length": 1}]))

    def test_icmp_without_ports_does_not_crash(self):
        pkts = pd.DataFrame([pk(0, 64, sport=None, dport=None, proto="ICMP"),
                             pk(1, 64, sport=None, dport=None, proto="ICMP")])
        flows = aggregate_packets(pkts)
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows.iloc[0].packet_count, 2)

    def test_bad_values_raise(self):
        with self.assertRaises(ValueError):
            aggregate_packets(pd.DataFrame([pk(0, -5)]))
        with self.assertRaises(ValueError):
            aggregate_packets(pd.DataFrame([pk(0, 5)]), idle_timeout=0)


if __name__ == "__main__":
    unittest.main()
