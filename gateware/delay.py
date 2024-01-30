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

# Delayer temporarily moved out of the sequencing module to break a circular dependency

from amaranth import Signal

from amaranth_cfu import SimpleElaboratable


class Delayer(SimpleElaboratable):
    """Delays an input Signal via a shift register.

    Parameters
    ----------
    cycles: int
        Number of cycles to delay the signal

    Public Interface
    ---------------
    input: Signal() in
        The input signal
    output: Signal() out
        Mirrors the input signal after cycles delay
    """

    def __init__(self):
        self.cycles = Signal(4)
        self.input = Signal()
        self.output = Signal()

    def elab(self, m):
        shift_register = Signal(6)

        with m.If(self.cycles==5):
            m.d.comb += self.output.eq(shift_register[4])
        with m.Elif(self.cycles==4):
            m.d.comb += self.output.eq(shift_register[3])
        with m.Elif(self.cycles==3):
            m.d.comb += self.output.eq(shift_register[2])  
        with m.Elif(self.cycles==2):
            m.d.comb += self.output.eq(shift_register[1])            
        with m.Else():
            m.d.comb += self.output.eq(shift_register[0])
        
        m.d.sync += [
            shift_register[1:].eq(shift_register),
            shift_register[0].eq(self.input),
        ]
