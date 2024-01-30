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


from amaranth.sim import Delay,Settle

from amaranth_cfu import TestBase

from mul import Mul8Pipeline
#,AccOrTrans, ByteToWordShifter


class Mul8PipelineTest(TestBase):
    def create_dut(self):
        return Mul8Pipeline ()

    def test(self):
        DATA = [
            ((10, 10, 0), 10),
            ((6, 25, 0), 10),
            ((23, 17, 0), 27),
            ((45, -29, 0), -2),
            ((-9, 25, 1), 23),
            ((9, 25, 0), 0),
            ((256, 10, 0), 10),
            ((73, 5, 1), 10),
            ((-82, 2, 0), 0),
        ]

        def process():
            #yield self.dut.offset.eq(2)
            '''
            for n, (inputs, expected) in enumerate(DATA):
                f_data, i_data, clear = inputs
                yield self.dut.f_data.eq(f_data)
                yield self.dut.i_data.eq(i_data)
                #yield Settle()
                #if expected is not None:
                    #self.assertEqual((yield self.dut.result), expected, f"case={n}")
                yield
                yield
            '''
        #self.run_sim(process, 0)


'''class AccOrTransTest(TestBase):
    def create_dut(self):
        return AccOrTrans()

    def test(self):
        DATA = [
            ((1, 10, 0), 10),
            ((0, 25, 0), 10),
            ((1, 17, 0), 27),
            ((1, -29, 0), -2),
            ((1, 25, 1), 23),
            ((0, 25, 0), 0),
            ((1, 10, 0), 10),
            ((0, 5, 1), 10),
            ((0, 2, 0), 0),
        ]

        def process():
            for n, (inputs, expected) in enumerate(DATA):
                add_en, in_value, clear = inputs
                yield self.dut.add_en.eq(add_en)
                yield self.dut.in_values.eq(in_value)
                yield self.dut.clear.eq(clear)
                yield Settle()
                #if expected is not None:
                    #self.assertEqual((yield self.dut.result), expected, f"case={n}")
                yield
                yield
        self.run_sim(process, 0)'''

'''
class ByteToWordShifterTest(TestBase):
    def create_dut(self):
        return ByteToWordShifter()

    def test(self):
        DATA = [
            ((0, 0), 0),
            ((1, 0), 0),
            ((0, 0), 0),
            ((1, 0x11), 0x11000000),
            ((0, 0), 0x11000000),
            ((1, 0x22), 0x22110000),
            ((1, 0x33), 0x33221100),
            ((0, 0), 0x33221100),
            ((1, 0x99), 0x99332211),
            ((1, 0x88), 0x88993322),
            ((1, 0x44), 0x44889933),
        ]

        def process():
            for n, (inputs, expected) in enumerate(DATA):
                shift_en, in_value = inputs
                yield self.dut.shift_en.eq(shift_en)
                yield self.dut.in_value.eq(in_value)
                yield Delay(0.25)
                if expected is not None:
                    self.assertEqual((yield self.dut.result), expected, f"case={n}")
                yield
        self.run_sim(process, 0)
'''


