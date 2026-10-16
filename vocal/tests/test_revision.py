import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from vocal_transcriber.config import Config, QuantizedNote
from vocal_transcriber.beat import BeatGrid
from vocal_transcriber.musicxml_export import export_musicxml
from vocal_transcriber.octaves import reconcile_octaves
from tests.test_core import track


class RevisionTests(unittest.TestCase):
    def test_naturals_context_and_ties(self):
        # Natural E crossing a bar, F#, F natural, repeated F natural.
        events = [QuantizedNote(p,s*.5,(s+d)*.5,s,d,.9) for p,s,d in
                  [(64,0,5),(66,5,1),(65,6,1),(65,7,1)]]
        grid = BeatGrid(120,np.array([0.,.5]),np.array([0.,1.]),'test')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.musicxml'
            export_musicxml(events,grid,Config(key='Am'),path)
            root=ET.parse(path)
            es=[n for n in root.findall('.//note') if n.findtext('pitch/step')=='E']
            self.assertTrue(all(n.find('accidental') is None for n in es))
            fs=[n for n in root.findall('.//note') if n.findtext('pitch/step')=='F']
            self.assertEqual([n.findtext('accidental') for n in fs],['sharp','natural',None])
            self.assertTrue(root.findall('.//tie'))

    def test_key_signature_cancellation(self):
        grid=BeatGrid(120,np.array([0.,.5]),np.array([0.,1.]),'test')
        events=[QuantizedNote(p,i*.5,(i+1)*.5,i,1,.9) for i,p in enumerate([66,65,66])]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.musicxml'
            export_musicxml(events,grid,Config(key='G'),path)
            root=ET.parse(path)
            self.assertEqual(root.findtext('.//fifths'),'1')
            self.assertEqual([n.findtext('accidental') for n in root.findall('.//note') if n.find('pitch') is not None],[None,'natural','sharp'])

    def compare(self, original, reference, frequency, confidence=.95):
        sr=16000
        t=np.arange(len(original)*160)/sr
        audio=.5*np.sin(2*np.pi*frequency*t)+.2*np.sin(4*np.pi*frequency*t)
        return reconcile_octaves(track(original),track(reference,np.full(len(original),confidence)),audio,sr,Config())

    def test_halved_f0_corrected_with_harmonic_evidence(self):
        result,audit=self.compare(np.full(100,57),np.full(100,69),440)
        np.testing.assert_allclose(result.midi_pitch,69)
        self.assertEqual(audit[0]['action'],'corrected')

    def test_real_low_note_not_raised_to_harmonic(self):
        result,audit=self.compare(np.full(100,45),np.full(100,57),110)
        np.testing.assert_allclose(result.midi_pitch,45)
        self.assertEqual(audit[0]['action'],'review')

    def test_strong_spectral_evidence_with_moderate_confidence(self):
        result,audit=self.compare(np.full(100,45),np.full(100,69),440,.62)
        np.testing.assert_allclose(result.midi_pitch,69)
        self.assertEqual(audit[0]['action'],'corrected')

    def test_agreement_and_unconfident_reference_unchanged(self):
        for original,reference,confidence in [(57,57,.95),(57,69,.3)]:
            result,_=self.compare(np.full(100,original),np.full(100,reference),220,confidence)
            np.testing.assert_allclose(result.midi_pitch,original)

    def test_non_octave_difference_unchanged(self):
        result,_=self.compare(np.full(100,57),np.full(100,64),330)
        np.testing.assert_allclose(result.midi_pitch,57)

    def test_confidence_dip_does_not_split_corrected_note(self):
        sr=16000
        t=np.arange(sr)/sr
        audio=.5*np.sin(2*np.pi*440*t)+.2*np.sin(2*np.pi*880*t)
        confidence=np.full(100,.95)
        confidence[40:55]=.2
        result,_=reconcile_octaves(track(np.full(100,57)),track(np.full(100,69),confidence),audio,sr,Config())
        np.testing.assert_allclose(result.midi_pitch,69)
