#coding:utf-8
import os
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)
import struct
import fuxi.common.SystemLog as Log

class Reader():
  ##KaldiArk.Reader constructor
  #@param scp_path path to the .scp file
  def __init__(self, scp_path):

    self.scp_position = 0
    fin = open(scp_path,"r")
    self.utt_ids = []
    self.scp_data = []
    line = fin.readline()
    while line != '' and line != None:
      utt_id, path_pos = line.replace('\n','').split(' ')
      path, pos = path_pos.split(':')
      self.utt_ids.append(utt_id)
      self.scp_data.append((path, int(pos)))
      line = fin.readline()

    fin.close()

  ## read data from the archive
  #@param index index of the utterance that will be read
  #@return a numpy array containing the data from the utterance
  def read_utt_data(self, index):
    ark_read_buffer = open(self.scp_data[index][0], 'rb')
    ark_read_buffer.seek(self.scp_data[index][1],0)
    header = struct.unpack(b'<xcccc', ark_read_buffer.read(5))
    if header[0] != b"B":
      Log.LogError("Input .ark file is not binary")
    if header[1] == b"C":
      Log.LogError("Input .ark file is compressed")

    rows = 0; cols= 0
    m, rows = struct.unpack(b'<bi', ark_read_buffer.read(5))
    n, cols = struct.unpack(b'<bi', ark_read_buffer.read(5))

    if header[1] == b"F":
      tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
    elif header[1] == b"D":
      tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8), dtype=np.float64)

    utt_mat = np.reshape(tmp_mat, (rows, cols))

    ark_read_buffer.close()

    return utt_mat

  ## read the next utterance in the scp file
  #@return the utterance ID of the utterance that was read, the utterance data, bool that is true if the reader looped back to the beginning
  def read_next_utt(self):

    if len(self.scp_data) == 0:
      return None , None, True

    if self.scp_position >= len(self.scp_data): #if at end of file loop around
      looped = True
      self.scp_position = 0
    else:
      looped = False

    self.scp_position += 1

    return self.utt_ids[self.scp_position-1], self.read_utt_data(self.scp_position-1), looped

  ## read the next utterance ID but don't read the data
  #@return the utterance ID of the utterance that was read
  def read_next_scp(self):

    if self.scp_position >= len(self.scp_data): #if at end of file loop around
      self.scp_position = 0

    self.scp_position += 1

    return self.utt_ids[self.scp_position-1]

  ## read the previous utterance ID but don't read the data
  #@return the utterance ID of the utterance that was read
  def read_previous_scp(self):

    if self.scp_position < 0: #if at beginning of file loop around
      self.scp_position = len(self.scp_data) - 1

    self.scp_position -= 1

    return self.utt_ids[self.scp_position+1]

  ## read the data of a certain utterance ID
  #@return the utterance data corresponding to the ID
  def read_utt(self, utt_id):

    return self.read_utt_data(self.utt_ids.index(utt_id))

  ##Split of the data that was read so far
  def split(self):
    self.scp_data = self.scp_data[self.scp_position:-1]
    self.utt_ids = self.utt_ids[self.scp_position:-1]

  @property
  def num_utt(self):
    return len(self.utt_ids)

class Writer():
  ##KaldiArk.Writer constructor
  #@param scp_path path to the .scp file that will be written
  def __init__(self, scp_path):
    self.scp_path = scp_path
    if self.scp_path != 'stdout':
      self.scp_file_write = open(self.scp_path,"w")
    else :
      self.scp_file_write = None
    ##read an utterance to the archive
    #@param ark_path path to the .ark file that will be used for writing
    #@param utt_id the utterance ID
    #@param utt_mat a numpy array containing the utterance data
  def write_next_utt(self, ark_path, utt_id, utt_mat):
    utt_mat = np.asarray(utt_mat, dtype=np.float32)
    rows, cols = utt_mat.shape
    if self.scp_file_write == None:
      ark_file_write = sys.stdout
      ark_file_write.buffer.write(struct.pack('<%dsc'%(len(utt_id)), utt_id, b' '))
      ark_file_write.buffer.write(struct.pack('<Bcccc',0x00,b'B',b'F',b'M',b' '))
      ark_file_write.buffer.write(struct.pack('<bi', 4, rows))
      ark_file_write.buffer.write(struct.pack('<bi', 4, cols))
      ark_file_write.buffer.write(utt_mat)
    else :
      ark_file_write = open(ark_path,"ab")
      ark_file_write.write(struct.pack('<%dsc'%(len(utt_id)), utt_id, b' '))
      pos = ark_file_write.tell()
      ark_file_write.write(struct.pack('<Bcccc',0x00, b'B',b'F',b'M',b' '))
      ark_file_write.write(struct.pack('<bi', 4, rows))
      ark_file_write.write(struct.pack('<bi', 4, cols))
      ark_file_write.write(utt_mat)
      self.scp_file_write.write('%s %s:%s\n' % (utt_id.decode('utf-8'), os.path.abspath(ark_path), pos))
      ark_file_write.close()

  ##close the ark writer
  def close(self):
    if self.scp_file_write != None:
      self.scp_file_write.close()
      self.scp_file_write = None
