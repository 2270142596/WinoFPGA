#!/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from amaranth_cfu import CfuTestBase, pack_vals

from amaranth import Elaboratable, Module, Mux, Signal, Memory
from amaranth.build import Platform
from amaranth.hdl.ast import Cat
from amaranth.sim import Simulator

import unittest

from mnv2_cfu import make_cfu


def pack_vals(*values, offset=0, bits=8):
    """Packs single values into a word in little endian order.
    offset is added to the values before packing
    bits is the number of bits in each value
    """
    mask = (1 << bits) - 1
    result = 0
    for i, v in enumerate(values):
        result += ((v + offset) & mask) << (i * bits)
    return result


class SimpleElaboratable(Elaboratable):
    """Simplified Elaboratable interface
    Widely, but not generally applicable. Suitable for use with
    straight-forward blocks of logic in a single domain.

    Attributes
    ----------

    m: Module
        The resulting module
    platform:
        The platform for this elaboration

    """

    def elab(self, m: Module):
        """Alternate elaborate interface"""
        return NotImplementedError()

    def elaborate(self, platform: Platform):
        self.m = Module()
        self.platform = platform
        self.elab(self.m)
        return self.m


class _PlaceholderSyncModule(SimpleElaboratable):
    """A module that does something arbirarty with synchronous logic

    This is used by TestBase to stop Amaranth from complaining if our DUT doesn't
    contain any synchronous logic."""

    def elab(self, m):
        state = Signal(1)
        m.d.sync += state.eq(~state)


class TestBase(unittest.TestCase):
    """Base class for testing an Amaranth module.

    The module can use sync, comb or both.
    """

    def setUp(self):
        # Create DUT and add to simulator
        self.m = Module()
        self.dut = self.create_dut()
        self.m.submodules['dut'] = self.dut
        self.m.submodules['placeholder'] = _PlaceholderSyncModule()
        self.sim = Simulator(self.m)

    def create_dut(self):
        """Returns an instance of the device under test"""
        raise NotImplementedError

    def add_process(self, process):
        """Add main test process to the simulator"""
        self.sim.add_sync_process(process)

    def add_sim_clocks(self):
        """Add clocks as required by sim.
        """
        self.sim.add_clock(1, domain='sync')

    def run_sim(self, process, write_trace=False):
        self.add_process(process)
        self.add_sim_clocks()
        if write_trace:
            with self.sim.write_vcd("zz.vcd", "zz.gtkw"):
                self.sim.run()
            # Discourage commiting code with tracing active
            self.fail("Simulation tracing active. "
                      "Turn off after debugging complete.")
        else:
            self.sim.run()


class CfuTestBase(TestBase):
    """Tests CFU ops independent of timing and handshaking."""

    def _unpack(self, inputs):
        return inputs if len(inputs) == 4 else (
            inputs[0], 0, inputs[1], inputs[2])

    def run_ops(self, data, write_trace=False):
        """Runs the given ops through the CFU, checking results.

        Arguments:
            data: [(input, expected_output)...]
                  if expected_output is None, it is ignored.
                  input is either a 3-tuple (function_id, in0, in1) or
                  a 4-tuple (function_id, funct7, in0, in1)
        """
        def process():
            for n, (inputs, expected) in enumerate(data):
                function_id, funct7, in0, in1 = self._unpack(inputs)
                # Set inputs and cmd_valid
                yield self.dut.cmd_function_id.eq((funct7 << 3) | function_id)
                yield self.dut.cmd_in0.eq(in0 & 0xffff_ffff)
                yield self.dut.cmd_in1.eq(in1 & 0xffff_ffff)
                yield self.dut.cmd_valid.eq(1)
                yield self.dut.rsp_ready.eq(1)
                yield
                # Wait until command accepted and response available
                while not (yield self.dut.cmd_ready):
                    yield
                yield self.dut.cmd_valid.eq(0)
                while not (yield self.dut.rsp_valid):
                    yield
                yield self.dut.rsp_ready.eq(0)

                # Ensure no errors, and output as expected
                if expected != 0:
                    actual = (yield self.dut.rsp_out)
                    # self.assertEqual(actual, expected & 0xffff_ffff,
                    #                  f"output {hex(actual)} != {hex(expected & 0xffff_ffff)} ::: " +
                    #                  f"function_id={function_id}, funct7={funct7}, " +
                    #                  f" in0={in0} {hex(in0)}, in1={in1} {hex(in1)} (n={n})")
                yield
            for n in range(20):
                yield
        self.run_sim(process, write_trace)


class CfuTest(CfuTestBase):
    def create_dut(self):
        return make_cfu()

    '''def test_simple(self):
        DATA = [
            #switch
            ((0, 35, 0, 0), 0),

            # Store output shift
            ((0, 22, 5, 0), 0),

            # Store filter value * 4
            ((0, 24, 666, 0), 0),
            ((0, 24, 777, 0), 0),
            ((0, 24, 888, 0), 0),
            ((0, 24, 999, 0), 0),

            # Get filter value * 5
            ((0, 110, 0, 0), 666),
            ((0, 110, 0, 0), 777),
            ((0, 110, 0, 0), 888),
            ((0, 110, 0, 0), 999),
            ((0, 110, 0, 0), 666),  # wrap around

            # Restart, store eight more filters, retrieve again
            ((0, 20, 8, 0), 0),
            ((0, 24, 111, 0), 0),
            ((0, 24, 222, 0), 0),
            ((0, 24, 333, 0), 0),
            ((0, 24, 444, 0), 0),
            ((0, 24, 555, 0), 0),
            ((0, 24, 666, 0), 0),
            ((0, 24, 777, 0), 0),
            ((0, 24, 888, 0), 0),
            ((0, 110, 0, 0), 111),
            ((0, 110, 0, 0), 222),
            ((0, 110, 0, 0), 333),
            ((0, 110, 0, 0), 444),
            ((0, 110, 0, 0), 555),
            ((0, 110, 0, 0), 666),
            ((0, 110, 0, 0), 777),
            ((0, 110, 0, 0), 888),
            ((0, 110, 0, 0), 111),
            ((0, 110, 0, 0), 222),
            ((0, 110, 0, 0), 333),
            ((0, 110, 0, 0), 444),

            # Store 4 filter value words
            ((0, 20, 4, 0), 8),
            ((0, 24, pack_vals(1, 2, 3, 4), 0), 0),
            ((0, 24, pack_vals(3, 3, 3, 3), 0), 0),
            ((0, 24, pack_vals(-128, -128, -128, -128), 0), 0),
            ((0, 24, pack_vals(-99, 92, -37, 2), 0), 0),

            # Store 4 input value words
            ((0, 10, 4, 0), 0),
            ((0, 25, pack_vals(2, 3, 4, 5, offset=-128), 0), 0),
            ((0, 25, pack_vals(2, 12, 220, 51, offset=-128), 0), 0),
            ((0, 25, pack_vals(255, 255, 255, 255, offset=-128), 0), 0),
            ((0, 25, pack_vals(19, 17, 103, 11, offset=-128), 0), 0),
        ]
        return self.run_ops(DATA, 0)
        '''
    '''
    def test_input_store(self):
        DATA = []

        def set_val(val):
            return ((0, 25, val, 0), 0)

        def get_val(val):
            return ((0, 111, 0, 0), val)

        def set_input_depth(val):
            return ((0, 10, val, 0), 0)

        def finish_read():
            return ((0, 112, 0, 0), 0)

        DATA = (
            [((0, 35, 0, 0), 0)] +
            [set_input_depth(10)] +
            [set_val(v) for v in range(100, 110)] +
            [get_val(v) for v in range(100, 110)] +
            [set_val(v) for v in range(200, 210)] +
            [get_val(v) for v in range(100, 110)] +
            [finish_read()] +
            [get_val(v) for v in range(200, 210)])
        return self.run_ops(DATA, 1)
'''

    def test_2dconv(self):
        # Run a whole pixel of data
        # 16 values in input (4 words)
        # 24 values in output
        # filter words = 16 * 24 / 4 = 96
        # Calculations are at
        # https://docs.google.com/spreadsheets/d/1tQeX8expePNFisVX0Jl5_ZmKCX1QHWVMGRlPebpmpas/edit
        def set_reg(reg, val):
            return ((0, reg, val, 0), 0)

        def set_out_channel_params(bias, mult, shift):
            yield set_reg(21, mult)
            yield set_reg(22, shift)
            yield set_reg(23, bias)

        def set_filter_val(val):
            return ((0, 24, val, 0), 0)

        def set_input_val(val):
            return ((0, 25, val, 0), 0)

        def get_output(expected_result):
            return ((0, 34, 0, 0), expected_result)

        def make_op_stream():
            def nums(start, count): return range(start, start + count)

            # switch=0
            yield set_reg(35, 0)

            # Output offset -50,
            yield set_reg(13, -128)
            # Input depth 4 words, input offset 50, batch size 24 outputs (6 words)
            yield set_reg(10, 4)
            yield set_reg(12, 128)
            yield set_reg(20, 24)
            # activation min max = -128, +127,
            yield set_reg(14, -128)
            yield set_reg(15, 127)

            for _ in range(6):
                yield from set_out_channel_params(30_000, 31_000_000, -3)
                yield from set_out_channel_params(50_000, 50_000_000, -6)
                yield from set_out_channel_params(75_000, 56_000_000, -4)
                yield from set_out_channel_params(100_000, 50_000_000, -5)

            for f_vals in zip(nums(-17, 96), nums(3, 96),
                              nums(-50, 96), nums(5, 96)):
                yield set_filter_val(pack_vals(*f_vals))

            for i_vals in zip(nums(1, 4), nums(3, 4), nums(5, 4), nums(7, 4)):
                yield set_input_val(pack_vals(*i_vals))

            # Start calculation
            yield set_reg(33, 0)
            yield get_output(pack_vals(-125, -117, -24, -57))
            yield get_output(pack_vals(-63, -105, 32, -32))
            yield get_output(pack_vals(-1, -92, 88, -7))
            yield get_output(pack_vals(60, -80, 127, 17))
            yield get_output(pack_vals(122, -67, 127, 42))
            yield get_output(pack_vals(127, -55, 127, 67))

            for i_vals in zip(nums(1, 4), nums(3, 4), nums(5, 4), nums(7, 4)):
                yield set_input_val(pack_vals(*i_vals))

            # Start calculation
            yield set_reg(33, 0)
            yield get_output(pack_vals(-125, -117, -24, -57))
            yield get_output(pack_vals(-63, -105, 32, -32))
            yield get_output(pack_vals(-1, -92, 88, -7))
            yield get_output(pack_vals(60, -80, 127, 17))
            yield get_output(pack_vals(122, -67, 127, 42))
            yield get_output(pack_vals(127, -55, 127, 67))

        return self.run_ops(make_op_stream(), 0)

    def test_depthwise_conv(self):
        # Run a whole pixel of data
        # 16 values in input (4 words)
        # 24 values in output
        # filter words = 16 * 24 / 4 = 96
        # Calculations are at
        # https://docs.google.com/spreadsheets/d/1tQeX8expePNFisVX0Jl5_ZmKCX1QHWVMGRlPebpmpas/edit
        def set_reg(reg, val):
            return ((0, reg, val, 0), 0)

        def set_out_channel_params(bias, mult, shift):
            yield set_reg(21, mult)
            yield set_reg(22, shift)
            yield set_reg(23, bias)

        def set_filter_val(val):
            return ((0, 24, val, 0), 0)

        def set_input_val(val):
            return ((0, 25, val, 0), 0)

        def get_output(expected_result):
            return ((0, 34, 0, 0), expected_result)

        def make_op_stream():

            # below is V include offset already

            def nums(start, count): return range(start, start + count)

            # switch=0
            yield set_reg(35, 0)

            i_d = 8
            o_d = 48

            # Output offset -50,
            yield set_reg(13, -128)
            # Input depth 4 words, input offset 50, batch size 24 outputs (6 words)
            yield set_reg(10, 4)
            yield set_reg(12, 128)

            # activation min max = -128, +127,
            yield set_reg(14, -128)
            yield set_reg(15, 127)
            yield set_reg(20, 24)

            for _ in range(6):
                yield from set_out_channel_params(30_000, 31_000_000, -3)
                yield from set_out_channel_params(50_000, 50_000_000, -6)
                yield from set_out_channel_params(75_000, 56_000_000, -4)
                yield from set_out_channel_params(100_000, 50_000_000, -5)

            for f_vals in zip(nums(-17, 96), nums(3, 96),
                              nums(-50, 96), nums(5, 96)):
                yield set_filter_val(pack_vals(*f_vals))

            for i_vals in zip(nums(1, 4), nums(3, 4), nums(5, 4), nums(7, 4)):
                yield set_input_val(pack_vals(*i_vals))

            # Start calculation
            yield set_reg(33, 0)
            yield get_output(pack_vals(-125, -117, -24, -57))
            yield get_output(pack_vals(-63, -105, 32, -32))
            yield get_output(pack_vals(-1, -92, 88, -7))
            yield get_output(pack_vals(60, -80, 127, 17))
            yield get_output(pack_vals(122, -67, 127, 42))
            yield get_output(pack_vals(127, -55, 127, 67))

            yield set_reg(35, 1)

            # num_tile=400
            # num_tile_half=200
            # num_tile_4=100
            # input_channel=48
            # input_channel_4=12
            input_width = 41
            pad = 1
            num_tile = 800
            # num_tile_half=50
            # num_tile_4=25
            input_channel = 16
            input_channel_2 = 8
            yield set_reg(36, num_tile)
            yield set_reg(37, input_width)

            # Output   -50,
            yield set_reg(13, -128)
            # Input depth 4 words, input offset 50, batch size 24 outputs (6 words)
            input_offset = 128
            yield set_reg(10,  21*(input_width+pad))
            yield set_reg(20, num_tile*4)
            yield set_reg(12, input_offset)
            # yield set_reg(20, 16*4)
            # activation min max = -128, +127,
            yield set_reg(14, -128)
            yield set_reg(15, 127)

            aa = [
                -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
                -128, -49, -46, -48, -50, -58, -63, -46, -22, -26, -39, -128,
                -128, -105, -88, -94, -94, -90, -76, -67, -36, -36, -58, -128,
                -128, -77, -82, -67, -45, -31, -24, -30, -33, -23, -50, -128,
                -128, -76, -46, -36, -39, -30, -42, -32, -26, -27, -54, -128,
                -128, -60, -30, -39, -35, -33, -42, -44, -38, -33, -53, -128,
                -128, -26, -35, -38, -34, -36, -40, -46, -39, -37, -49, -128,
                -128, -29, -33, -39, -33, -44, -42, -47, -31, -31, -37, -128,
                -128, -39, -41, -33, -43, -40, -41, -30, -33, -33, -45, -128,
                -128, -52, -54, -58, -70, -44, -50, -56, -54, -55, -79, -128,
                -128, -30, -30, -27, -27, -29, -31, -32, -30, -26, -46, -128,
                -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            ]
            bb = [
                -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
                -128, 88, 96, 94, 94, 95, 98, 86, 71, 72, 34, -128,
                -128, -28, -3, 1, -1, -16, -8, 0, -40, -36, -69, -128,
                -128, -62, -51, -45, -51, -46, -65, -60, -70, -92, -112, -128,
                -128, -50, -34, -44, -70, -68, -83, -70, -76, -91, -88, -128,
                -128, -87, -82, -79, -84, -80, -84, -86, -68, -75, -76, -128,
                -128, -87, -82, -73, -74, -98, -81, -79, -76, -78, -85, -128,
                -128, -65, -57, -52, -72, -79, -70, -72, -70, -87, -85, -128,
                -128, -81, -73, -61, -73, -75, -85, -61, -92, -98, -86, -128,
                -128, -14, -7, -11, -1, -15, -12, -4, -2, 0, -6, -128,
                -128, 55, 63, 61, 61, 60, 58, 63, 63, 60, 28, -128,
                -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128]
# channel 1:a a a a; channel 2: b b ;
            for i in range(input_channel_2):

                yield from set_out_channel_params(-285, 1272858496, -6)
                yield from set_out_channel_params(411,1124309888,-6)

# set num_tile= (pow(Input depth ,0.5)-2)/2,but now num_tile=4
            for i in range(input_channel_2):

                yield set_filter_val(pack_vals(7, -10, 2, 127))  # (0 1 2 3)
                yield set_filter_val(pack_vals(-25, -103, 12, 35))  # (0 1 2 3)
                yield set_filter_val(pack_vals(-39, 0, 0, 0))  # (0 1 2 3)
                
                yield set_filter_val(pack_vals(31, 127, 26, -17))  # (0 1 2 3)
                yield set_filter_val(pack_vals(-37, -13, -12, -106))  # (0 1 2 3)
                yield set_filter_val(pack_vals(-7, 0, 0, 0))  # (0 1 2 3)

            for j in range(21):
                for i in range(input_width):
                    yield set_input_val(pack_vals(aa[0], aa[1], aa[12], aa[13]))
                for k in range(pad):
                    yield set_input_val(pack_vals(0, 0, 0, 0))

            # Start calculation
            yield set_reg(33, 0)

            for j in range(num_tile):
                yield get_output(pack_vals(100, 100, 101, 101))  # n=42

            for j in range(21):
                for i in range(input_width):
                    yield set_input_val(pack_vals(bb[0], bb[1], bb[12], bb[13]))
                for k in range(pad):
                    yield set_input_val(pack_vals(0, 0, 0, 0))

            # Start calculation
            yield set_reg(33, 0)


            for j in range(num_tile):
                yield get_output(pack_vals(100, 100, 101, 101))  # n=42

        return self.run_ops(make_op_stream(), 1)


'''class mnv2Test(unittest.TestCase):
    def setUp(self):               ### (D)
        self.m = Module()
        self.dut = CfuTest().create_dut()
        self.m.submodules['dut'] = self.dut

    def test_simple_edge(self):
        def process():            ### (E)
            # TODO: Add test code here
            m.submodules['dut'].test_2dconv()
            yield
        self.run_sim(process)

    ### (F)
    def run_sim(self, process, write_trace=True):
        sim = Simulator(self.m)
        sim.add_sync_process(process)
        ###sim.add_clock(1)
        if write_trace:
            with sim.write_vcd("zz.vcd", "zz.gtkw"):
                sim.run()
        else:
            sim.run()

if __name__ == '__main__':
    unittest.main()
'''
