import random
import midi
import datetime
from collections import Counter

class Markov(object):
	"""docstring for Markov"""
	def __init__(self, order=2):
		super(Markov, self).__init__()
		self.order = order
		self.chain = {}

	def add(self,key,value):
		if key in self.chain:
			self.chain[key].append(value)
		else:
			self.chain[key] = [value]

	def load(self,midifile):
		pattern = midi.read_midifile(midifile)
		#print(pattern[1])
		#track = pattern[1]
		for track in pattern:
			noteslist = []
			curoffset = 0
			for i in track:
				#print(i)
				if i.name == "Note On" and i.data[1]!=0:
					'''data[0]--note_num ; data[1]--velocity ; tick+curoffset--interval between each onset'''
					note = (i.data[0],i.data[1],i.tick+curoffset)
					#note = (i.data[0],i.data[1],i.tick)
					noteslist.append(note)
					curoffset = 0
				else:
					curoffset+=i.tick
			if len(noteslist)>self.order:
				'''The concept of order: http://web.ntnu.edu.tw/~algo/HiddenMarkovModel.html'''
				for j in range(self.order,len(noteslist)):
					t = tuple(noteslist[j-self.order:j])
					#print('t', t)
					#print('noteslist[j]', noteslist[j])
					self.add(t,noteslist[j])
			else:
				print("Corpus too short")
		
	def generate(self, length, filename):
		pattern = midi.Pattern()
		# Instantiate a MIDI Track (contains a list of MIDI events)
		track = midi.Track()
		# Append the track to the pattern
		pattern.append(track)

		tick = 0
		currenttuple = random.choice(list(self.chain.keys()))
		prevnote = False
		for i in range(0,self.order):
			if prevnote!=False:
				on = midi.NoteOnEvent(tick=tick, velocity=0, pitch=prevnote, channel=9)
				track.append(on)
			on = midi.NoteOnEvent(tick=0, velocity=currenttuple[i][1], pitch=currenttuple[i][0], channel=9)
			track.append(on)
			tick = currenttuple[i][2]
			prevnote = currenttuple[i][0]
		result = random.choice(self.chain[currenttuple])
		#counts = Counter(self.chain[currenttuple])
		#result = counts.most_common(1)[0][0]
		for i in range(1,length):
			for j in range(0,self.order):
				if prevnote!=False:
					if tick>5000:
						tick=5000
					on = midi.NoteOnEvent(tick=tick, velocity=0, pitch=prevnote, channel=9)
					track.append(on)
				on = midi.NoteOnEvent(tick=0, velocity=currenttuple[j][1], pitch=currenttuple[j][0], channel=9)
				track.append(on)
				tick = currenttuple[j][2]
				prevnote = currenttuple[j][0]

			currenttuple = list(currenttuple)
			currenttuple.pop(0)
			currenttuple.append(result)
			currenttuple = tuple(currenttuple)
			if currenttuple in self.chain:
				result = random.choice(self.chain[currenttuple])
				#counts = Counter(self.chain[currenttuple])
				#result = counts.most_common(1)[0][0]
			else:
				result = random.choice(self.chain[random.choice(list(self.chain.keys()))])
				#counts = Counter(self.chain[random.choice(list(self.chain.keys()))])
				#result = counts.most_common(1)[0][0]
				
		# Add the end of track event, append it to the track
		eot = midi.EndOfTrackEvent(tick=1)
		track.append(eot)
		# Print out the pattern
		#print('pattern', pattern)
		# Save the pattern to disk
		midi.write_midifile(filename, pattern)

def markov_drum(order, input_path, output_path):
	m = Markov(order) # set order
	print("Loading music")
	try:
		m.load(input_path)
	except Exception as e:
		print("File not found or corrupt")
	m.generate(1000, output_path)
	print("Done")


def main():
	# m = Markov(order)  ### Set order
	print("Loading music")
	inp = input('Name of midi file to load or g to generate: ')
	name = inp
	markov_drum(3, 'train/' + name, name)
	# while inp != "g":
	# 	try:
	# 		print('load!')
	# 		m.load('train/'+inp)
	# 	except Exception as e:
	# 		print("File not found or corrupt")
	# 	inp = input('Name of midi file to load or g to generate: ')
	# print ("Done")
	# m.generate(1000,name)

if __name__ == '__main__':
	main()