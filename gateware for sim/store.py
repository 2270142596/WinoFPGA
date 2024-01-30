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

from amaranth import Array, Signal, Mux, Cat, signed

from amaranth_cfu import SimpleElaboratable, is_pysim_run, DualPortMemory, SequentialMemoryReader, tree_sum

# from .registerfile import Xetter
# from .sequencing import UpCounter

from registerfile import Xetter
from sequencing import UpCounter


class StoreSetter(Xetter):
    """Stores in0 into a Store.

    Parameters
    ----------
    bits_width:
        bits in each word of memory
    num_memories:
        number of memories in the store. data will be striped over the memories.
        Assumed to be a power of two.
    depth:
        Maximum number of items stored in each memory.

    Public Interface
    ----------------
    w_en: Signal(1)[num_memories] output
        Memory write enable.
    w_addr: Signal(range(depth)) output
        Memory address to which to write
    w_data: Signal(width) output
        Data to write
    restart: Signal input
        Signal to drop all parameters from memory and restart all counters.
    count: Signal(range(depth*num_memories+1)) output
        How many items the memory currently holds
    updated: Signal() output
        Indicates that store has been updated with a new value or restarted
    """

    def __init__(self, bits_width, num_memories, depth):
        super().__init__()
        assert 0 == ((num_memories - 1) &
                     num_memories), "Num memories (f{num_memories}) not a power of two"
        self.num_memories_bits = (num_memories - 1).bit_length()
        self.w_en = Array(Signal() for _ in range(num_memories))
        self.w_addr = Signal(range(depth))
        self.w_data = Signal(bits_width)
        self.restart = Signal()
        self.count = Signal(range(depth * num_memories + 1))
        self.updated = Signal()

    def connect_write_port(self, dual_port_memories):
        """Connects the write port of a list of dual port memories to this.

        Returns a list of statements that perform the connection.
        """
        assert len(dual_port_memories) == len(
            self.w_en), f"Memory length does not match: {dual_port_memories}, {len(self.w_en)}"
        statement_list = []
        for w_en, dp in zip(self.w_en, dual_port_memories):
            statement_list.extend([
                dp.w_en.eq(w_en),
                dp.w_addr.eq(self.w_addr),
                dp.w_data.eq(self.w_data),
            ])
        return statement_list

    def elab(self, m):
        m.d.comb += [
            self.done.eq(True),
            self.w_addr.eq(self.count[self.num_memories_bits:]),
            self.updated.eq(Cat(self.w_en).any() | self.restart),
        ]
        with m.If(self.restart):
            m.d.sync += self.count.eq(0)
        with m.Elif(self.start):
            m.d.comb += [
                self.w_en[self.count[:self.num_memories_bits]].eq(1),
                self.w_data.eq(self.in0),
            ]
            m.d.sync += self.count.eq(self.count + 1)


class CircularIncrementer(SimpleElaboratable):
    """Does circular increments of a memory address counter over a single memory
    range.

    Parameters
    ----------
    depth:
        Maximum number of items stored in memory

    Public Interface
    ----------------
    restart: Signal input
        Signal to reset address to zero
    next: Signal input
        Produce next piece of data (available next cycle).
    limit: Signal(range(depth))) input
        Number of items stored in memory
    r_addr: Signal(range(depth)) output
        Next address
    """

    def __init__(self, depth):
        self.depth = depth
        self.restart = Signal()
        self.next = Signal()
        self.limit = Signal(range(depth))
        self.r_addr = Signal(range(depth))

    def elab(self, m):
        # Current r_addr is the address being presented to memory this cycle
        # By default current address is last address, but this can be
        # modied by the restart or next signals
        last_addr = Signal.like(self.r_addr)
        m.d.sync += last_addr.eq(self.r_addr)
        m.d.comb += self.r_addr.eq(last_addr)

        # Respond to inputs
        with m.If(self.restart):
            m.d.comb += self.r_addr.eq(0)
        with m.Elif(self.next):
            m.d.comb += self.r_addr.eq(Mux(last_addr >=
                                           self.limit - 1, 0, last_addr + 1))


class FilterValueFetcher(SimpleElaboratable):
    """Fetches next single word from a 4-way filter value store.

    Parameters
    ----------
    max_depth:
        Maximum number of items stored in each memory

    Public Interface
    ----------------
    limit: Signal(range(depth)) input
        Number of entries currently contained in the filter store, divided by 4
    mem_addr: Signal(range(depth/num_memories))[4] output
        Current read pointer for each memory
    mem_data: Signal(range(depth/num_memories))[4] input
        Current value being read from each memory
    data: Signal(32)[4] output
        Four words being fetched
    next: Signal() input
        Indicates that fetched value has been read.
    updated: Signal() input
        Indicates that memory store has been updated, and processing should start at start
    restart: Signal() input
        Soft reset signal to restart all processing
    """

    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.limit = Signal(range(max_depth * 4))
        self.mem_addrs = [
            Signal(
                range(max_depth),
                name=f"mem_addr_{n}") for n in range(4)]
        self.mem_datas = [Signal(32, name=f"mem_data_{n}") for n in range(4)]
        self.data = Signal(192)
        self.next = Signal()
        self.updated = Signal()
        self.restart = Signal()
        self.switch_PorD = Signal()
        

        self.smrs = [
            SequentialMemoryReader(
                width=32,
                max_depth=max_depth) for _ in range(4)]

    def connect_read_ports(self, dual_port_memories):
        """Helper method to connect a list of dual port memories to self.

        Returns a list of statements that perform the connection.
        """
        result = []
        for mem_addr, mem_data, dpm in zip(
                self.mem_addrs, self.mem_datas, dual_port_memories):
            result += [
                dpm.r_addr.eq(mem_addr),
                mem_data.eq(dpm.r_data),
            ]
        return result

    def elab(self, m):
        was_updated = Signal()
        m.d.sync += was_updated.eq(self.updated)
        # Connect sequential memory readers
        smr_datas = Array([Signal(32, name=f"smr_data_{n}") for n in range(4)])
        smr_nexts = Array([Signal(name=f"smr_next_{n}") for n in range(4)])
        for n, (smr, mem_addr, mem_data, smr_data, smr_next) in enumerate(
                zip(self.smrs, self.mem_addrs, self.mem_datas, smr_datas, smr_nexts)):
            m.submodules[f"smr_{n}"] = smr
            m.d.comb += [
                smr.limit.eq((self.limit + 3 - n) >> 2),
                mem_addr.eq(smr.mem_addr),
                smr.mem_data.eq(mem_data),
                smr_data.eq(smr.data),
                smr.next.eq(smr_next),
                smr.restart.eq(was_updated),
            ]

# add addr

        r_addr = Signal.like(self.limit)
        # curr_bank = Signal(2)
        # m.d.sync += self.data.eq(Cat(smr_datas[r_addr[:2]], smr_datas[r_addr[:2]+1],
        #                          smr_datas[r_addr[:2]+2], smr_datas[r_addr[:2]+3]))
        
        tmp = Signal(128)
        data_tmp=[Signal(signed(12), name=f"data_tmp_{n}") for n in range(16)]
        late_tmp=[Signal(signed(8), name=f"late_tmp_{n}") for n in range(4)]
        m.d.comb += tmp.eq(Cat(smr_datas[r_addr[:2]], smr_datas[(r_addr+1)[:2]],
                           smr_datas[(r_addr+2)[:2]]))
        def cut(l, no, width):
                return l[no*width:(no+1)*width]
        with m.If(self.switch_PorD == 0):
            m.d.sync += [cut(self.data, n, 12).eq((cut(tmp, n, 8).as_signed())) for n in range(8)
                         ]
        with m.Else():
            m.d.sync += [data_tmp[0].eq((cut(tmp, 0, 8).as_signed())<<2),
                         data_tmp[1].eq(tree_sum(iter([cut(tmp, 0, 8).as_signed(),cut(tmp, 1, 8).as_signed(),cut(tmp, 2, 8).as_signed()]))),
                         data_tmp[2].eq(tree_sum(iter([cut(tmp, 0, 8).as_signed(),cut(tmp, 3, 8).as_signed(),cut(tmp, 6, 8).as_signed()]))),
                         data_tmp[3].eq((tree_sum(iter([cut(tmp, 0, 8).as_signed(),cut(tmp, 3, 8).as_signed(),cut(tmp, 6, 8).as_signed(),cut(tmp, 1, 8).as_signed(),cut(tmp, 4, 8).as_signed()]))).as_signed()),
                         data_tmp[4].eq(tree_sum(iter([cut(tmp, 0, 8).as_signed(),-cut(tmp, 1, 8).as_signed(),cut(tmp, 2, 8).as_signed()]))),
                         data_tmp[5].eq((cut(tmp, 2, 8).as_signed())<<2),
                         data_tmp[6].eq((tree_sum(iter([cut(tmp, 0, 8).as_signed(),cut(tmp, 3, 8).as_signed(),cut(tmp, 6, 8).as_signed(),-cut(tmp, 1, 8).as_signed(),-cut(tmp, 4, 8).as_signed()]))).as_signed()),
                         data_tmp[7].eq(tree_sum(iter([cut(tmp, 2, 8).as_signed(),cut(tmp, 5, 8).as_signed(),cut(tmp, 8, 8).as_signed()]))),
                         data_tmp[8].eq(tree_sum(iter([cut(tmp, 0, 8).as_signed(),-cut(tmp, 3, 8).as_signed(),cut(tmp, 6, 8).as_signed()]))),
                         data_tmp[9].eq((tree_sum(iter([cut(tmp, 0, 8).as_signed(),-cut(tmp, 3, 8).as_signed(),cut(tmp, 6, 8).as_signed(),cut(tmp, 1, 8).as_signed(),-cut(tmp, 4, 8).as_signed()]))).as_signed()),
                         data_tmp[10].eq((cut(tmp, 6, 8).as_signed())<<2),
                         data_tmp[11].eq(tree_sum(iter([cut(tmp, 6, 8).as_signed(),cut(tmp, 7, 8).as_signed(),cut(tmp, 8, 8).as_signed()]))),
                         data_tmp[12].eq((tree_sum(iter([cut(tmp, 0, 8).as_signed(),-cut(tmp, 3, 8).as_signed(),cut(tmp, 6, 8).as_signed(),-cut(tmp, 1, 8).as_signed(),cut(tmp, 4, 8).as_signed()]))).as_signed()),
                         data_tmp[13].eq(tree_sum(iter([cut(tmp, 2, 8).as_signed(),-cut(tmp, 5, 8).as_signed(),cut(tmp, 8, 8).as_signed()]))),
                         data_tmp[14].eq(tree_sum(iter([cut(tmp, 6, 8).as_signed(),-cut(tmp, 7, 8).as_signed(),cut(tmp, 8, 8).as_signed()]))),
                         data_tmp[15].eq((cut(tmp, 8, 8).as_signed())<<2),
                
                        

                
                         
                         late_tmp[0].eq(cut(tmp, 7, 8).as_signed()),
                         late_tmp[1].eq(cut(tmp, 2, 8).as_signed()),
                         late_tmp[2].eq(cut(tmp, 5, 8).as_signed()),
                         late_tmp[3].eq(cut(tmp, 8, 8).as_signed()),


                         cut(self.data, 0, 12).eq(data_tmp[0]),
                         cut(self.data, 1, 12).eq((data_tmp[1]) << 1),
                         cut(self.data, 2, 12).eq((data_tmp[2]) << 1),
                         cut(self.data, 3, 12).eq((tree_sum(iter(
                             [data_tmp[3], late_tmp[0], late_tmp[1], late_tmp[2], late_tmp[3]]))).as_signed()),
                         cut(self.data, 4, 12).eq((data_tmp[4]) << 1),
                         cut(self.data, 5, 12).eq(data_tmp[5]),
                         cut(self.data, 6, 12).eq((tree_sum(iter(
                             [data_tmp[6], -late_tmp[0], late_tmp[1], late_tmp[2], late_tmp[3]]))).as_signed()),
                         cut(self.data, 7, 12).eq(data_tmp[7] << 1),
                         cut(self.data, 8, 12).eq(data_tmp[8] << 1),
                         cut(self.data, 9, 12).eq((tree_sum(iter(
                             [data_tmp[9], late_tmp[0], late_tmp[1], -late_tmp[2], late_tmp[3]]))).as_signed()),
                         cut(self.data, 10, 12).eq(data_tmp[10]),
                         cut(self.data, 11, 12).eq(data_tmp[11] << 1),
                         cut(self.data, 12, 12).eq((tree_sum(iter(
                             [data_tmp[12], -late_tmp[0], late_tmp[1], -late_tmp[2], late_tmp[3]]))).as_signed()),
                         cut(self.data, 13, 12).eq(data_tmp[13] << 1),
                         cut(self.data, 14, 12).eq(data_tmp[14] << 1),
                         cut(self.data, 15, 12).eq(data_tmp[15]),
                        ]

            # m.d.sync += [cut(self.data, 0, 12).eq((cut(tmp, 0, 8).as_signed()).as_signed()<<2),
            #              cut(self.data, 1, 12).eq((cut(tmp, 0, 8).as_signed()+cut(tmp, 1, 8).as_signed()+cut(tmp, 2, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 2, 12).eq((cut(tmp, 0, 8).as_signed()+cut(tmp, 3, 8).as_signed()+cut(tmp, 6, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 3, 12).eq((cut(tmp, 0, 8).as_signed()+cut(tmp, 3, 8).as_signed()+cut(tmp, 6, 8).as_signed()+cut(tmp, 1, 8).as_signed()+cut(tmp, 4, 8).as_signed()+cut(tmp, 7, 8).as_signed()+cut(tmp, 2, 8).as_signed()+cut(tmp, 5, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()),
            #              cut(self.data, 4, 12).eq((cut(tmp, 0, 8).as_signed()-cut(tmp, 1, 8).as_signed()+cut(tmp, 2, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 5, 12).eq((cut(tmp, 2, 8).as_signed()).as_signed()<<2),
            #              cut(self.data, 6, 12).eq((cut(tmp, 0, 8).as_signed()+cut(tmp, 3, 8).as_signed()+cut(tmp, 6, 8).as_signed()-cut(tmp, 1, 8).as_signed()-cut(tmp, 4, 8).as_signed()-cut(tmp, 7, 8).as_signed()+cut(tmp, 2, 8).as_signed()+cut(tmp, 5, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()),
            #              cut(self.data, 7, 12).eq((cut(tmp, 2, 8).as_signed()+cut(tmp, 5, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 8, 12).eq((cut(tmp, 0, 8).as_signed()-cut(tmp, 3, 8).as_signed()+cut(tmp, 6, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 9, 12).eq((cut(tmp, 0, 8).as_signed()-cut(tmp, 3, 8).as_signed()+cut(tmp, 6, 8).as_signed()+cut(tmp, 1, 8).as_signed()-cut(tmp, 4, 8).as_signed()+cut(tmp, 7, 8).as_signed()+cut(tmp, 2, 8).as_signed()-cut(tmp, 5, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()),
            #              cut(self.data, 10, 12).eq((cut(tmp, 6, 8).as_signed()).as_signed()<<2),
            #              cut(self.data, 11, 12).eq((cut(tmp, 6, 8).as_signed()+cut(tmp, 7, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 12, 12).eq((cut(tmp, 0, 8).as_signed()-cut(tmp, 3, 8).as_signed()+cut(tmp, 6, 8).as_signed()-cut(tmp, 1, 8).as_signed()+cut(tmp, 4, 8).as_signed()-cut(tmp, 7, 8).as_signed()+cut(tmp, 2, 8).as_signed()-cut(tmp, 5, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()),
            #              cut(self.data, 13, 12).eq((cut(tmp, 2, 8).as_signed()-cut(tmp, 5, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 14, 12).eq((cut(tmp, 6, 8).as_signed()-cut(tmp, 7, 8).as_signed()+cut(tmp, 8, 8).as_signed()).as_signed()<<1),
            #              cut(self.data, 15, 12).eq((cut(tmp, 8, 8).as_signed()).as_signed()<<2),
            #              ]
            
        # a=Signal(12)
        # b = [Signal(8) for n in range(4)]
        # m.d.comb+=[b[0].eq(cut(tmp, 2, 8)),
        # b[1].eq(cut(tmp, 2, 8)),
        # b[2].eq(cut(tmp, 5, 8)),
        # b[3].eq(cut(tmp, 13, 8))]
        # m.d.sync +=a.eq(b[0].as_signed()<<2)
        # c=Signal(12)
        # d=Signal(8)
        # m.d.sync +=c.eq((cut(tmp, 1, 8).as_signed()).as_signed()<<2)
        # m.d.sync +=d.eq(-b[1].as_signed())


        with m.If(self.restart):
            m.d.sync += r_addr.eq(0)
        with m.Elif(self.next):
            with m.If(self.switch_PorD == 0):
                m.d.comb += smr_nexts[r_addr[:2]].eq(1)
                m.d.comb += smr_nexts[r_addr[:2]+1].eq(1)

                with m.If(r_addr == self.limit - 2):
                    m.d.sync += r_addr.eq(0)
                with m.Else():
                    m.d.sync += r_addr.eq(r_addr + 2)
            with m.Else():
                m.d.comb += smr_nexts[r_addr[:2]].eq(1)
                m.d.comb += smr_nexts[(r_addr+1)[:2]].eq(1)
                m.d.comb += smr_nexts[(r_addr+2)[:2]].eq(1)
                # m.d.comb += smr_nexts[r_addr[:2]+3].eq(1)

                with m.If(r_addr == self.limit - 3):
                    m.d.sync += r_addr.eq(0)
                with m.Else():
                    m.d.sync += r_addr.eq(r_addr + 3)

        '''curr_bank = Signal(2)
        m.d.comb += self.data.eq(smr_datas[curr_bank])

        with m.If(self.restart):
            m.d.sync += curr_bank.eq(0)
        with m.Elif(self.next):
            m.d.comb += smr_nexts[curr_bank].eq(1)
            m.d.sync += curr_bank.eq(curr_bank + 1)
            '''


class NextWordGetter(Xetter):
    """Gets the next word from a store.

    Public Interface
    ----------------
    data: Signal(32) input
        The current value to be fetched
    next: Signal() output
        Indicates that fetched value has been read.
    ready: Signal() input
        Signal from the store that data is valid. The read only completes when ready is true.
    """

    def __init__(self):
        super().__init__()
        self.data = Signal(32)
        self.next = Signal()
        self.ready = Signal()

    def elab(self, m):
        waiting = Signal()
        with m.If(self.ready & (waiting | self.start)):
            m.d.comb += [
                self.output.eq(self.data),
                self.next.eq(1),
                self.done.eq(1),
            ]
            m.d.sync += waiting.eq(0)
        with m.Elif(self.start & ~self.ready):
            m.d.sync += waiting.eq(1)


class InputStore(SimpleElaboratable):
    """Stores one "pixel" of input values for processing.

    This store is double-buffered so that processing may continue
    on one buffer while data is being loaded onto the other.

    All channels for a single pixel of data are written once, then
    read many times. Reads are performed circularly over the input
    data.

    Attributes
    ----------
    max_depth: int
        maximum allowed input depth.
        Assumed to be power of two.

    Public Interface
    ----------------
    restart: Signal() input
        Resets component to known state to begin processing.
    input_depth: Signal(range(max_depth * 4 // 2)) input
        Number of words per input pixel

    w_data: Signal(32) input
        Data to store
    w_en: Signal() input
        Hold high to store data - w_ready must also be high for data
        to be considered stored.
    w_ready: Signal() output
        Indicates that store is read to receive data.

    r_ready: Signal() output
        Indicates that data has been stored and reading is allowed
    r_data: Signal(32) output
        Four words of data
    r_next: Signal() input
        Data being read this cycle, so advance data pointer by one word next cycle
    r_finished: Signal() input
        Last data read this cycle, so move to next buffer on next cycle
    """

    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.restart = Signal()
        self.input_depth = Signal(range(max_depth * 4 // 2))
        self.w_data = Signal(32)
        self.w_en = Signal()
        self.w_ready = Signal()
        self.r_ready = Signal()
        self.r_data = Signal(192)  # TODO: increase to 2 and 4 words
        self.r_next = Signal()
        self.r_finished = Signal()
        self.switch_PorD = Signal()
        self.offset = Signal(signed(12))
        self.input_width=Signal(32)
        self.pad=Signal(4)

    def _elab_read(self, m, dps, r_full):
        r_curr_buf = Signal()
        # Address within current buffer
        r_addr = Signal.like(self.input_depth)
        m.submodules['input_width_count'] = input_width_count = UpCounter(8)
        m.d.comb += [
            input_width_count.max.eq(self.input_width-1),
            input_width_count.restart.eq(self.restart|self.r_finished),
            input_width_count.en.eq(self.r_next & self.r_ready),
        ]
        
        # Create and connect sequential memory readers
        smrs = [
            SequentialMemoryReader(
                max_depth=self.max_depth // 2,
                width=32) for _ in range(4)]
        smr_nexts = Array(smr.next for smr in smrs)
        mem_addrs =Array(Signal(range(self.max_depth // 2),  name=f"mem_addrs_{n}") for n in range(4))
        for (n, dp, smr,mem_addr) in zip(range(4), dps, smrs,mem_addrs):
            m.submodules[f"smr_{n}"] = smr
            m.d.comb += [
                #dp.r_addr.eq(Cat(smr.mem_addr, r_curr_buf)),##edit switch

                smr.mem_data.eq(dp.r_data),
                smr.limit.eq((self.input_depth + 3 - n) >> 2),
            ]
            with m.If(self.switch_PorD == 0):
                m.d.comb +=dp.r_addr.eq(Cat(smr.mem_addr, r_curr_buf))
            with m.Else():
                m.d.comb +=dp.r_addr.eq(Cat(mem_addr, r_curr_buf))



        # Ready if current buffer is full
        full = Signal()
        m.d.comb += full.eq(r_full[r_curr_buf])
        last_full = Signal()
        m.d.sync += last_full.eq(full)


        with m.If(full & ~last_full):
            m.d.sync += self.r_ready.eq(1)
            with m.If(self.switch_PorD == 0):
                m.d.comb += [smr.restart.eq(1) for smr in smrs]
            with m.Else():
                m.d.sync += r_addr.eq(1)
                m.d.comb +=input_width_count.en.eq(1)



        # Increment address
        # edit r_addr

        with m.If(self.r_next & self.r_ready):
            with m.If(self.switch_PorD == 0):
                with m.If(r_addr == self.input_depth - 2):
                    m.d.sync += r_addr.eq(0)
                with m.Else():
                    m.d.sync += r_addr.eq(r_addr + 2)

                m.d.comb += smr_nexts[r_addr[:2]].eq(1)
                m.d.comb += smr_nexts[r_addr[:2]+1].eq(1)
            with m.Else():
                with m.If(input_width_count.done==0):
                    m.d.sync += r_addr.eq(r_addr+1)
                with m.Else():
                    m.d.sync += r_addr.eq(r_addr + self.pad+2)

                m.d.comb += mem_addrs[(r_addr[:2])].eq((r_addr[2:]))
                m.d.comb += mem_addrs[((r_addr+1)[:2])].eq(((r_addr+1)[2:]))
                m.d.comb += mem_addrs[((r_addr+self.input_width+self.pad)[:2])].eq(((r_addr+self.input_width+self.pad)[2:]))
                m.d.comb += mem_addrs[((r_addr+self.input_width+self.pad+1)[:2])].eq(((r_addr+self.input_width+self.pad+1)[2:]))
        with m.Else():
            m.d.comb += mem_addrs[0].eq(0)
            m.d.comb += mem_addrs[1].eq(0)
            m.d.comb += mem_addrs[2].eq(((self.input_width+self.pad)[2:]))
            m.d.comb += mem_addrs[3].eq(((self.input_width+self.pad+1)[2:]))



  

        # Get data
        smr_datas = Array(smr.data for smr in smrs)
        smr_mem_datas = Array(smr.mem_data for smr in smrs)

        # tmp = Signal(128)
        # r_data_tmp=[Signal(signed(12), name=f"data_tmp_{n}") for n in range(16)]
        # m.d.comb += tmp.eq(Cat(smr_datas[r_addr[:2]], smr_datas[r_addr[:2]+1],
        #                    smr_datas[r_addr[:2]+2], smr_datas[r_addr[:2]+3]))
        


        tmp = Signal(128)

        tmp1 = Signal(128)

        tmp2 = Signal(128)
        r_data_tmp=[Signal(signed(12), name=f"data_tmp_{n}") for n in range(16)]

        # m.d.comb += tmp1.eq(Cat(smr_datas[r_addr[:2]], smr_datas[r_addr[:2]+1],
        #                    smr_datas[r_addr[:2]+2], smr_datas[r_addr[:2]+3]))
        # m.d.comb += tmp2.eq(Cat(smr_mem_datas[(r_addr-1)[:2]], smr_mem_datas[(r_addr)[:2]],
        #                    smr_mem_datas[(r_addr+1)[:2]], smr_mem_datas[(r_addr+2)[:2]]))
        # with m.If(r_addr[:2]==0):
        #     m.d.comb += tmp1.eq(Cat(smr_datas[0], smr_datas[1]))
        # with m.Elif(r_addr[:2]==2):
        #     m.d.comb += tmp1.eq(Cat(smr_datas[2], smr_datas[3]))
        smr_no_p=Signal(2)
        m.d.comb+=smr_no_p.eq(r_addr[:2])
        with m.If(smr_no_p==0):
            m.d.comb += tmp1.eq(Cat(smr_datas[0], smr_datas[1]))
        with m.Elif(smr_no_p==2):
            m.d.comb += tmp1.eq(Cat(smr_datas[2], smr_datas[3]))

        smr_no_d=Signal(2)
        with m.If(self.r_next & self.r_ready):
            m.d.sync+=smr_no_d.eq(r_addr[:2])
        with m.Else():
            m.d.sync+=smr_no_d.eq(0)

        with m.If(smr_no_d==0):
            m.d.comb += tmp2.eq(Cat(smr_mem_datas[0], smr_mem_datas[1],
                           smr_mem_datas[2], smr_mem_datas[3]))            
        with m.Elif(smr_no_d==1):
            m.d.comb += tmp2.eq(Cat(smr_mem_datas[1], smr_mem_datas[2],
                           smr_mem_datas[3], smr_mem_datas[0]))
        with m.Elif(smr_no_d==2):
            m.d.comb += tmp2.eq(Cat(smr_mem_datas[2], smr_mem_datas[3],
                           smr_mem_datas[0], smr_mem_datas[1]))
        with m.Elif(smr_no_d==3):
            m.d.comb += tmp2.eq(Cat(smr_mem_datas[3], smr_mem_datas[0],
                           smr_mem_datas[1], smr_mem_datas[2]))
        # with m.If(r_addr[:2]==0):
        #     m.d.comb += tmp1.eq(Cat(smr_datas[0], smr_datas[1]))
        #     m.d.comb += tmp2.eq(Cat(smr_mem_datas[3], smr_mem_datas[0],
        #                    smr_mem_datas[1], smr_mem_datas[2]))
        # with m.Elif(r_addr[:2]==1):
        #     m.d.comb += tmp2.eq(Cat(smr_mem_datas[0], smr_mem_datas[1],
        #                    smr_mem_datas[2], smr_mem_datas[3]))
        # with m.Elif(r_addr[:2]==2):
        #     m.d.comb += tmp1.eq(Cat(smr_datas[2], smr_datas[3]))
        #     m.d.comb += tmp2.eq(Cat(smr_mem_datas[1], smr_mem_datas[2],
        #                    smr_mem_datas[3], smr_mem_datas[0]))
        # with m.Elif(r_addr[:2]==3):
        #     m.d.comb += tmp2.eq(Cat(smr_mem_datas[2], smr_mem_datas[3],
        #                    smr_mem_datas[0], smr_mem_datas[1]))


        with m.If(self.switch_PorD == 0):
            m.d.comb += tmp.eq(tmp1)
        with m.Else():
            m.d.comb += tmp.eq(tmp2)

        # m.d.comb += tmp1.eq(Cat(smr_datas[0], smr_datas[1],
        #                    smr_datas[2], smr_datas[3]))
        # m.d.comb += tmp2.eq(Cat(smr_mem_datas[0], smr_mem_datas[1],
        #                    smr_mem_datas[2], smr_mem_datas[3]))
        # with m.If(self.switch_PorD == 0):
        #     m.d.comb += tmp.eq(tmp1>>((r_addr[1:2])*32))
        # with m.Else():
        #     m.d.comb += tmp.eq(tmp2>>(((r_addr-1)[:2])*32))

        


        def cut(l, no, width,offset=0):
                return l[no*width:(no+1)*width]
        



        with m.If(self.switch_PorD == 0):
            m.d.sync += [cut(self.r_data, n, 12).eq((cut(tmp, n, 8).as_signed())+self.offset) for n in range(8)
                         ]
        with m.Else():

            m.d.sync += [
                         r_data_tmp[0].eq((tree_sum(iter([cut(tmp, 0, 8).as_signed(),-cut(tmp, 8, 8).as_signed()-cut(tmp, 4, 8).as_signed(),cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[1].eq((tree_sum(iter([cut(tmp, 1, 8).as_signed(),-cut(tmp, 9, 8).as_signed(),cut(tmp, 4, 8).as_signed(),-cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[2].eq((tree_sum(iter([ cut(tmp, 2, 8).as_signed(),cut(tmp, 8, 8).as_signed()-cut(tmp, 6, 8).as_signed(),-cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[3].eq((tree_sum(iter([cut(tmp, 3, 8).as_signed(),cut(tmp, 9, 8).as_signed(),cut(tmp, 6, 8).as_signed(),cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[4].eq((tree_sum(iter([-cut(tmp, 1, 8).as_signed(),cut(tmp, 9, 8).as_signed(),cut(tmp, 4, 8).as_signed(),-cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[5].eq((tree_sum(iter([cut(tmp, 1, 8).as_signed(),-cut(tmp, 9, 8).as_signed(),-cut(tmp, 5, 8).as_signed(),cut(tmp, 13, 8).as_signed()]))).as_signed()),
                         r_data_tmp[6].eq((tree_sum(iter([-cut(tmp, 3, 8).as_signed(),-cut(tmp,9, 8).as_signed(),cut(tmp, 6, 8).as_signed(),cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[7].eq((tree_sum(iter([cut(tmp, 3, 8).as_signed(),cut(tmp, 9, 8).as_signed(),-cut(tmp, 7, 8).as_signed(),-cut(tmp, 13, 8).as_signed()]))).as_signed()),
                         r_data_tmp[8].eq((tree_sum(iter([-cut(tmp, 2, 8).as_signed(),cut(tmp, 8, 8).as_signed(),cut(tmp, 6, 8).as_signed(),-cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[9].eq((tree_sum(iter([-cut(tmp, 3, 8).as_signed(),cut(tmp,9, 8).as_signed(),-cut(tmp, 6, 8).as_signed(),cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[10].eq((tree_sum(iter([cut(tmp, 2, 8).as_signed(),-cut(tmp, 10, 8).as_signed(),-cut(tmp, 6, 8).as_signed(),cut(tmp, 14, 8).as_signed()]))).as_signed()),
                         r_data_tmp[11].eq((tree_sum(iter([cut(tmp, 3, 8).as_signed(),-cut(tmp, 11, 8).as_signed(),cut(tmp, 6, 8).as_signed(),-cut(tmp, 14, 8).as_signed()]))).as_signed()),
                         r_data_tmp[12].eq((tree_sum(iter([cut(tmp, 3, 8).as_signed(),-cut(tmp, 9, 8).as_signed(),-cut(tmp, 6, 8).as_signed(),cut(tmp, 12, 8).as_signed()]))).as_signed()),
                         r_data_tmp[13].eq((tree_sum(iter([-cut(tmp, 3, 8).as_signed(),cut(tmp,9, 8).as_signed(),cut(tmp, 7, 8).as_signed(),-cut(tmp, 13, 8).as_signed()]))).as_signed()),
                         r_data_tmp[14].eq((tree_sum(iter([-cut(tmp, 3, 8).as_signed(),cut(tmp,11, 8).as_signed(),cut(tmp, 6, 8).as_signed(),-cut(tmp, 14, 8).as_signed()]))).as_signed()),
                         r_data_tmp[15].eq((tree_sum(iter([cut(tmp, 3, 8).as_signed(),-cut(tmp, 11, 8).as_signed(),-cut(tmp, 7, 8).as_signed(),cut(tmp, 15, 8).as_signed()]))).as_signed()),
                
                


                         cut(self.r_data, 0, 12).eq(r_data_tmp[0]),
                         cut(self.r_data, 1, 12).eq(r_data_tmp[1]),
                         cut(self.r_data, 2, 12).eq(r_data_tmp[2]),
                         cut(self.r_data, 3, 12).eq((tree_sum(iter([ r_data_tmp[3],(self.offset<<2)]))).as_signed()),
                         cut(self.r_data, 4, 12).eq(r_data_tmp[4]),
                         cut(self.r_data, 5, 12).eq(r_data_tmp[5]),
                         cut(self.r_data, 6, 12).eq(r_data_tmp[6]),
                         cut(self.r_data, 7, 12).eq(r_data_tmp[7]),
                         cut(self.r_data, 8, 12).eq(r_data_tmp[8]),
                         cut(self.r_data, 9, 12).eq(r_data_tmp[9]),
                         cut(self.r_data, 10, 12).eq(r_data_tmp[10]),
                         cut(self.r_data, 11, 12).eq(r_data_tmp[11]),
                         cut(self.r_data, 12, 12).eq(r_data_tmp[12]),
                         cut(self.r_data, 13, 12).eq(r_data_tmp[13]),
                         cut(self.r_data, 14, 12).eq(r_data_tmp[14]),
                         cut(self.r_data, 15, 12).eq(r_data_tmp[15]),
                         
                         ]


        # On finished (overrides r_next)
        with m.If(self.r_finished):
            m.d.sync += [
                r_addr.eq(0),
                r_full[r_curr_buf].eq(0),
                last_full.eq(0),
                self.r_ready.eq(0),
                r_curr_buf.eq(~r_curr_buf),
            ]

        # On restart, reset addresses and Sequential Memory readers, go to not
        # ready
        with m.If(self.restart):
            m.d.sync += [
                r_addr.eq(0),
                last_full.eq(0),
                self.r_ready.eq(0),
                r_curr_buf.eq(0),
            ]

    def _elab_write(self, m, dps, r_full):
        w_curr_buf = Signal()
        # Address within current buffer
        w_addr = Signal.like(self.input_depth)

        # Connect to memory write port
        for n, dp in enumerate(dps):
            m.d.comb += [
                dp.w_addr.eq(Cat(w_addr[2:], w_curr_buf)),
                dp.w_data.eq(self.w_data),
                dp.w_en.eq(self.w_en & self.w_ready & (n == w_addr[:2])),
            ]
        # Ready to write current buffer if reading is not allowed
        m.d.comb += self.w_ready.eq(~r_full[w_curr_buf])

        # Write address increment
        with m.If(self.w_en & self.w_ready):
            with m.If(w_addr == self.input_depth - 1):
                # at end of buffer - mark buffer ready for reading and go to
                # next
                m.d.sync += [
                    w_addr.eq(0),
                    r_full[w_curr_buf].eq(1),
                    w_curr_buf.eq(~w_curr_buf),
                ]
            with m.Else():
                m.d.sync += w_addr.eq(w_addr + 1)

        with m.If(self.restart):
            m.d.sync += [
                w_addr.eq(0),
                w_curr_buf.eq(0),
            ]

    def elab(self, m):
        # Create the four memories
        dps = [
            DualPortMemory(
                depth=self.max_depth,
                width=32,
                is_sim=is_pysim_run()) for _ in range(4)]
        for n, dp in enumerate(dps):
            m.submodules[f"dp_{n}"] = dp

        # Tracks which buffers are "full" and ready for reading
        r_full = Array([Signal(name="r_full_0"), Signal(name="r_full_1")])
        with m.If(self.restart):
            m.d.sync += [
                r_full[0].eq(0),
                r_full[1].eq(0),
            ]

        # Track reading and writing
        self._elab_write(m, dps, r_full)
        self._elab_read(m, dps, r_full)


class InputStoreSetter(Xetter):
    """Puts a word into the input store.

    Public Interface
    ----------------
    w_data: Signal(32) input
        Data to send to input value store
    w_en: Signal() output
        Indicate ready to write data
    w_ready: Signal() input
        Indicates input store is ready to receive data
    """

    def __init__(self):
        super().__init__()
        self.w_data = Signal(32)
        self.w_en = Signal()
        self.w_ready = Signal()

    def connect(self, input_store):
        """Connect to self to input_store.

        Returns a list of statements that performs the connection.
        """
        return [
            input_store.w_data.eq(self.w_data),
            input_store.w_en.eq(self.w_en),
            self.w_ready.eq(input_store.w_ready),
        ]

    def elab(self, m):
        buffer = Signal(32)
        waiting = Signal()

        with m.If(self.start & self.w_ready):
            m.d.comb += [
                self.done.eq(1),
                self.w_en.eq(1),
                self.w_data.eq(self.in0)
            ]
        with m.Elif(self.start & ~self.w_ready):
            m.d.sync += [
                waiting.eq(1),
                buffer.eq(self.in0),
            ]
        with m.Elif(waiting & ~self.w_ready):
            m.d.comb += [
                self.done.eq(1),
                self.w_en.eq(1),
                self.w_data.eq(self.in0)
            ]
            m.d.sync += waiting.eq(1)