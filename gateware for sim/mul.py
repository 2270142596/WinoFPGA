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

from amaranth import Cat, Signal, signed

from amaranth_cfu import all_words, SimpleElaboratable, tree_sum


# from .delay import Delayer
# from .post_process import PostProcessor
# from .registerfile import Xetter

from delay import Delayer
from post_process import PostProcessor
from registerfile import Xetter


class Mul8Pipeline(SimpleElaboratable):
    """A 8-wide Multiply Add pipeline.

    Pipeline takes 2 additional cycles.

    f_data and i_data each contain 4 signed 8 bit values. The
    calculation performed is:

    result = sum((i_data[n] + offset) * f_data[n] for n in range(4))

    Public Interface
    ----------------
    offset: Signal(signed(8)) input
        Offset to be added to all inputs.
    f_data: Signal(32) input
        8 bytes of filter data to use next
    i_data: Signal(32) input
        8 bytes of input data data to use next
    result: Signal(signed(32)) output
        Result of the multiply and add
    """
    PIPELINE_CYCLES_P = 3  # 1 for x,1 for add in acc
    PIPELINE_CYCLES_D = 1

    def __init__(self):
        super().__init__()
        self.offset = Signal(signed(9))
        self.f_data = Signal(192)
        self.i_data = Signal(192)
        self.switch_PorD = Signal()
        # self.result = Signal(signed(32))
        # self.products = [Signal(signed(17), name=f"product_{n}") for n in range(4)]
        self.products = [Signal(signed(24)) for n in range(16)]

    def elab(self, m):

        # Product is 17 bits: 8 bits * 9 bits = 17 bits
        # products = [Signal(signed(17), name=f"product_{n}") for n in range(4)]

        for i_val, f_val, product in zip(
                all_words(self.i_data, 12), all_words(self.f_data, 12), self.products):

            f_tmp = Signal(signed(12))
            m.d.sync += f_tmp.eq(f_val.as_signed())
            i_tmp = Signal(signed(12))

            m.d.sync += i_tmp.eq(i_val.as_signed())
            m.d.comb += product.eq(i_tmp * f_tmp)

        # m.d.sync += self.result.eq(tree_sum(self.products))


class AccOrTrans(SimpleElaboratable):
    """An accumulator for a Mul8Pipline

    Public Interface
    ----------------
    add_en: Signal() input
        When to add the input
    in_value: Signal(signed(32)) input
        The input data to add
    clear: Signal() input
        Zero accumulator.
    result: Signal(signed(32)) output
        Result of the multiply and add
    """

    def __init__(self):
        super().__init__()
        self.add_en = Signal()
        self.in_values = [Signal(signed(24)) for n in range(16)]
        self.switch_PorD = Signal()
        self.clear = Signal()
        self.result = [Signal(signed(32), reset=0x0,
                              name=f"result_{n}") for n in range(4)]

    def elab(self, m):
        with m.If(self.switch_PorD == 0):
            # for pointwise
            add_sum = Signal(signed(28))
            m.d.sync += add_sum.eq(tree_sum(self.in_values[n]
                                   for n in range(8)))
            accumulator = Signal(signed(32))
            # m.d.comb += self.result.eq(accumulator)
            with m.If(self.add_en):
                m.d.sync += accumulator.eq(accumulator + add_sum)
                m.d.comb += self.result[0].eq(accumulator + add_sum)
            with m.Else():
                m.d.comb += self.result[0].eq(accumulator)
            # clear always resets accumulator next cycle, even if add_en is high
            # late_clear=Signal()
            # m.d.sync+=late_clear.eq(self.clear)
            with m.If(self.clear):
                m.d.sync += accumulator.eq(0)

        with m.Else():
            # for depthwise
            # clear always resets accumulator next cycle, even if add_en is high
            # late_clear=Signal()
            # m.d.sync+=late_clear.eq(self.clear)
            Y = [Signal(signed(32), reset=0x0,
                        name=f"Y_{n}") for n in range(4)]
            Y_tmp = [Signal(signed(32), reset=0x0,
                            name=f"Y_tmp_{n}") for n in range(4)]
            
            late_in_values=[Signal(signed(24), reset=0x0,
                            name=f"late_in_values_{n}") for n in range(4)]
            m.d.sync += [late_in_values[0].eq(self.in_values[12]),
                         late_in_values[1].eq(self.in_values[13]),
                         late_in_values[2].eq(self.in_values[14]),
                         late_in_values[3].eq(self.in_values[15]),



            ]
            # A = Signal(signed(32))
            # b = Signal(signed(32))
            # m.d.sync += A.eq(0xFFFFF008)
            # m.d.sync +=b.eq(A>>2)

            m.d.sync += Y_tmp[0].eq(tree_sum(iter([self.in_values[0], self.in_values[2], self.in_values[8], self.in_values[1],
                                    self.in_values[3], self.in_values[9], self.in_values[4],
                                    self.in_values[6]])))
            m.d.sync += Y_tmp[1].eq(tree_sum(iter([self.in_values[1], self.in_values[3], self.in_values[9], -self.in_values[4],
                                    -self.in_values[6], -self.in_values[12], -self.in_values[5], -
                                    self.in_values[7]])))
            m.d.sync += Y_tmp[2].eq(tree_sum(iter([self.in_values[2], -self.in_values[8], -self.in_values[10], self.in_values[3],
                                    -self.in_values[9], - self.in_values[11], self.in_values[6], -
                                    self.in_values[12]])))
            m.d.sync += Y_tmp[3].eq(tree_sum(iter([self.in_values[3], -self.in_values[9], -self.in_values[11], -self.in_values[6],
                                    self.in_values[12], self.in_values[14], -
                                                   self.in_values[7],
                                    self.in_values[13]])))

            m.d.sync += Y[0].eq((Y_tmp[0]+late_in_values[0]) )
            m.d.sync += Y[1].eq((Y_tmp[1]-late_in_values[1]) )
            m.d.sync += Y[2].eq((Y_tmp[2]-late_in_values[2]) )
            m.d.sync += Y[3].eq((Y_tmp[3]+late_in_values[3]) )

            # m.d.sync += Y[0].eq((self.in_values[0]+self.in_values[1]+self.in_values[2]+self.in_values[3] +
            #                               self.in_values[4]+self.in_values[6]+self.in_values[8]+self.in_values[9]+self.in_values[12])>>2)
            # m.d.sync += Y[1].eq((self.in_values[1]+self.in_values[3]-self.in_values[4]-self.in_values[5] -
            #                               self.in_values[6]-self.in_values[7]+self.in_values[9]-self.in_values[12]-self.in_values[13])>>2)
            # m.d.sync += Y[2].eq((self.in_values[2]+self.in_values[3]+self.in_values[6]-self.in_values[8] -
            #                               self.in_values[9]-self.in_values[10]-self.in_values[11]-self.in_values[12]-self.in_values[14])>>2)
            # m.d.sync += Y[3].eq((self.in_values[3]-self.in_values[6]-self.in_values[7]-self.in_values[9] -
            #                               self.in_values[11]+self.in_values[12]+self.in_values[13]+self.in_values[14]+self.in_values[15])>>2)
            m.d.comb += self.result[0].eq(Y[0]>>2)
            m.d.comb += self.result[1].eq(Y[1]>>2)
            m.d.comb += self.result[2].eq(Y[2]>>2)
            m.d.comb += self.result[3].eq(Y[3]>>2)


class ByteToWordShifter(SimpleElaboratable):
    """Shifts bytes into a word.

    Bytes are shifted from high to low, so that result is little-endian,
    with the "first" byte occupying the LSBs

    Public Interface
    ----------------
    shift_en: Signal() input
        When to shift the input
    in_value: Signal(8) input
        The input data to shift
    result: Signal(32) output
        Result of the shift
    """

    def __init__(self):
        super().__init__()
        self.shift_en = Signal()
        self.in_value = Signal(8)
        self.clear = Signal()
        self.result = Signal(32)

    def elab(self, m):
        register = Signal(32)
        m.d.comb += self.result.eq(register)

        with m.If(self.shift_en):
            calc = Signal(32)
            m.d.comb += [
                calc.eq(Cat(register[8:], self.in_value)),
                self.result.eq(calc),
            ]
            m.d.sync += register.eq(calc)
