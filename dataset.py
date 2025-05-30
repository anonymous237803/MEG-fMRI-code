import torch
from torch.utils.data import Dataset
import mne
import pickle
import numpy as np
from utils import flatten_recursive, zscore_tensor, nan_zscore
from sklearn.decomposition import PCA


def create_file_path_dict(
    MEG_SUBJECT,
    FMRI_SUBJECT,
    all_stories,
    ds_freq=50,
    spacing="oct6",
    WORD="semantic_c20_layer7",
    PHONEME="phoneme",
    SPECTRUM="melspectrum",
    MEG_VERSION="sss-S_band-1-150_notch-60-120_ecg-ica_eog-ica_audio-corrected",
    DATASET_DIR="/data/story_dataset",
    STIM_DIR="design_matrix",
    MEG_DIR="stacked_ridge_preds",
    FMRI_DIR="src_fmri",
    simulate=False,
    meg_cnr=0,
    fmri_cnr=0,
):
    # story to session and block mapping
    with open("data/story_sess_block.pkl", "rb") as f:
        story_sess_block = pickle.load(f)
    # story to unique story mapping
    with open("data/story_to_uniquestory.pkl", "rb") as f:
        story_to_uniquestory = pickle.load(f)

    file_path_dict = {}
    
    # design matrices
    LOC_WORD = f"{DATASET_DIR}/{STIM_DIR}/ds{ds_freq}/{WORD}/{MEG_SUBJECT}_stretched_avgrate.pkl"
    LOC_PHONEME = f"{DATASET_DIR}/{STIM_DIR}/ds{ds_freq}/{PHONEME}/{MEG_SUBJECT}_stretched_avgrate.pkl"
    LOC_SPECTRUM = f"{DATASET_DIR}/{STIM_DIR}/ds{ds_freq}/{SPECTRUM}/{MEG_SUBJECT}_stretched_avgrate.pkl"
    file_path_dict["word"] = LOC_WORD
    file_path_dict["phoneme"] = LOC_PHONEME
    file_path_dict["spectrum"] = LOC_SPECTRUM
    
    # meg and fmri
    file_path_dict["meg"], file_path_dict["fmri"] = {}, {}
    if simulate == False:
        LOC_MEG_TEMPLATE = f"{DATASET_DIR}/{MEG_DIR}/{{}}/{MEG_VERSION}/{MEG_SUBJECT}/aligned_ds{ds_freq}/{MEG_SUBJECT}_{{}}_{{}}_{MEG_VERSION}_aligned_ds{ds_freq}_raw.fif"
        LOC_FMRI_TEMPLATE = f"{DATASET_DIR}/{FMRI_DIR}/{spacing}/{FMRI_SUBJECT}_{{}}"
        for story in all_stories:
            session, block = story_sess_block[story]
            story_unique = story_to_uniquestory[story]
            file_path_dict["meg"][story] = LOC_MEG_TEMPLATE.format(session, session, block)
            file_path_dict["fmri"][story] = LOC_FMRI_TEMPLATE.format(story_unique)
    elif simulate == True:
        LOC_MEG_TEMPLATE = f"{DATASET_DIR}/{MEG_DIR}/{MEG_SUBJECT}_{spacing}_{{}}_{meg_cnr}_meg-raw.fif"
        LOC_FMRI_TEMPLATE = f"{DATASET_DIR}/{FMRI_DIR}/{MEG_SUBJECT}_{spacing}_{{}}_{fmri_cnr}_fmri"
        for story in all_stories:
            story_unique = story_to_uniquestory[story]
            file_path_dict["meg"][story] = LOC_MEG_TEMPLATE.format(story_unique)
            file_path_dict["fmri"][story] = LOC_FMRI_TEMPLATE.format(story_unique)
        print("Loading simulation data!")
        
    return file_path_dict


class StoryDataset(Dataset):
    def __init__(
        self,
        MEG_SUBJECT,
        FMRI_SUBJECT,
        all_stories,
        name="WdPnFq",
        ds_freq=50,
        tr=2,
        window_t=10,
        spacing="oct6",
        pca_meg=None,
        pca_mri=None,
        preload=True,
        **kwargs,
    ):
        self.name = name
        self.MEG_SUBJECT = MEG_SUBJECT
        self.FMRI_SUBJECT = FMRI_SUBJECT
        self.all_stories = all_stories
        self.all_stories_flatten = flatten_recursive(all_stories)
        self.ds_freq = ds_freq
        self.tr = tr
        self.window = window_t * ds_freq
        self.spacing = spacing

        # flags
        self.use_word = "Wd" in name
        self.use_phoneme = "Pn" in name
        self.use_freq = "Fq" in name
        self.use_meg = "Meg" in name
        self.use_mri = "Mri" in name
        print(f"use_word: {self.use_word}, use_phoneme: {self.use_phoneme}, use_freq: {self.use_freq}, use_meg: {self.use_meg}, use_mri: {self.use_mri}")

        # build file-path dict once
        self.file_path_dict = create_file_path_dict(
            MEG_SUBJECT=MEG_SUBJECT,
            FMRI_SUBJECT=FMRI_SUBJECT,
            all_stories=self.all_stories_flatten,
            ds_freq=ds_freq,
            spacing=spacing,
            **kwargs,
        )

        # PCA setup if needed
        if self.use_meg:
            self.pca_meg = pca_meg or self._pca_on_meg()
        else:
            self.pca_meg = None
        if self.use_mri:
            self.pca_mri = pca_mri or self._pca_on_mri()
        else:
            self.pca_mri = None

        # figure out embed_dim
        self.embed_dim = 0
        if self.use_word:
            self.embed_dim += 768
        if self.use_phoneme:
            self.embed_dim += 44
        if self.use_freq:
            self.embed_dim += 40
        if self.use_meg:
            self.embed_dim += self.pca_meg.n_components_
        if self.use_mri:
            self.embed_dim += self.pca_mri.n_components_
        print("embed_dim: ", self.embed_dim)

        # how many TRs we padded for fMRI-embed alignment
        self._pad_tr = 5

        # preload
        self.preload = preload
        if self.preload:
            self._preload()
            print("Preloaded all stories!")

    def __len__(self):
        return len(self.all_stories)

    def __getitem__(self, idx):
        story = self.all_stories[idx]
        if isinstance(story, list):  # if story is a list of stories, we average them
            embeds, meg, fmri = [], [], []
            for s in story:
                if self.preload:
                    embeds_, meg_, fmri_ = self.story_data[s]["embeds"], self.story_data[s]["meg"], self.story_data[s]["fmri"]
                else:
                    embeds_, meg_, fmri_ = self._load_story_data(s)
                embeds.append(embeds_)
                meg.append(meg_)
                fmri.append(fmri_)
            embeds = torch.mean(torch.stack(embeds), dim=0)
            meg = torch.mean(torch.stack(meg), dim=0)
            fmri = torch.mean(torch.stack(fmri), dim=0)
        else:  # if story is a single story
            if self.preload:
                embeds, meg, fmri = self.story_data[story]["embeds"], self.story_data[story]["meg"], self.story_data[story]["fmri"]
            else:
                embeds, meg, fmri = self._load_story_data(story)

        return embeds, meg, fmri

    def _preload(self):
        self.story_data = {}
        for story in self.all_stories_flatten:
            embeds, meg, fmri = self._load_story_data(story)
            self.story_data[story] = {"embeds": embeds, "meg": meg, "fmri": fmri}

    def _load_story_data(self, story):
        print(f"Loading story {story}...")

        # 1) design mats
        mats = []
        if self.use_word:
            with open(self.file_path_dict["word"], "rb") as f:
                mats.append(pickle.load(f)[story])
        if self.use_phoneme:
            with open(self.file_path_dict["phoneme"], "rb") as f:
                mats.append(pickle.load(f)[story])
        if self.use_freq:
            with open(self.file_path_dict["spectrum"], "rb") as f:
                mats.append(pickle.load(f)[story])

        # trim to shortest and concat
        min_len = min(m.shape[0] for m in mats)
        mats = [m[:min_len] for m in mats]
        embeds = torch.from_numpy(np.concatenate(mats, axis=1)).float()

        # 2) MEG
        meg = mne.io.read_raw_fif(self.file_path_dict["meg"][story], verbose=False).pick("meg").load_data(verbose=False).get_data().T
        meg = torch.from_numpy(meg).float()
        meg = zscore_tensor(meg, dim=0)

        # 3) fMRI
        stc_fmri = mne.read_source_estimate(self.file_path_dict["fmri"][story])
        fmri = stc_fmri.data.T
        fmri = torch.from_numpy(fmri).float()
        fmri = zscore_tensor(fmri, dim=0)

        # 4) align lengths to fMRI
        n_tr = fmri.shape[0]
        T = (n_tr + self._pad_tr) * self.tr * self.ds_freq
        while embeds.shape[0] < T or meg.shape[0] < T:
            n_tr -= 1
            T = (n_tr + self._pad_tr) * self.tr * self.ds_freq
        fmri = fmri[:n_tr]
        embeds = embeds[:T]
        meg = meg[:T]

        # 5) add MEG-PCA features
        if self.use_meg:
            meg_pc = self.pca_meg.transform(meg.numpy())
            meg_pc = torch.from_numpy(meg_pc).float()
            embeds = torch.cat([embeds, meg_pc], dim=1)

        # 6) add fMRI-PCA features into embed
        if self.use_mri:  # pad 5 TR of zeros, then upsample by ds_freq
            fmri_pc = self.pca_mri.transform(fmri.numpy())
            fmri_pc = torch.from_numpy(fmri_pc).float()
            fmri_pc = torch.cat([torch.zeros(5, fmri_pc.shape[1]), fmri_pc], dim=0)
            fmri_pc = fmri_pc.repeat_interleave(self.ds_freq * self.tr, dim=0)
            embeds = torch.cat([embeds, fmri_pc], dim=1)

        print(f"Finished loading story {story}!")

        return embeds, meg, fmri

    def _pca_on_meg(self, n_components=0.99):
        meg_list = []
        for story in self.all_stories_flatten:
            meg_path = self.file_path_dict["meg"][story]
            meg = mne.io.read_raw_fif(meg_path, verbose=False).pick("meg").load_data(verbose=False).get_data().T
            meg = nan_zscore(meg, axis=0)
            meg_list.append(meg)
        all_meg = np.concatenate(meg_list, axis=0)
        print(f"meg for PCA shape: {all_meg.shape}")
        pca = PCA(n_components=n_components).fit(all_meg)
        print("n components: ", pca.n_components_)
        print("meg pca explained variance ratio: ", np.sum(pca.explained_variance_ratio_))
        return pca

    def _pca_on_mri(self, n_components=0.99):
        fmri_list = []
        for story in self.all_stories_flatten:
            fmri_path = self.file_path_dict["fmri"][story]
            stc_fmri = mne.read_source_estimate(fmri_path)
            fmri = stc_fmri.data.T
            fmri = nan_zscore(fmri, axis=0)
            fmri_list.append(fmri)
        all_fmri = np.concatenate(fmri_list, axis=0)
        print(f"fmri for PCA shape: {all_fmri.shape}")
        pca = PCA(n_components=n_components).fit(all_fmri)
        print("n components: ", pca.n_components_)
        print("fmri pca explained variance ratio: ", np.sum(pca.explained_variance_ratio_))
        return pca


# --- story in pieces --- #

class StorySegmentDataset(Dataset):
    def __init__(
        self,
        MEG_SUBJECT,
        FMRI_SUBJECT,
        all_stories,
        name="WdPnFq",
        ds_freq=50,
        tr=2,
        window_t=10,
        seg_sec=40,
        overlap_sec=20,
        spacing="oct6",
        pca_meg=None,
        pca_mri=None,
        **kwargs,
    ):
        self.name = name
        self.MEG_SUBJECT = MEG_SUBJECT
        self.FMRI_SUBJECT = FMRI_SUBJECT
        self.all_stories = all_stories
        self.ds_freq = ds_freq
        self.tr = tr
        self.window = window_t * ds_freq
        self.spacing = spacing

        # feature flags
        self.use_word = "Wd" in name
        self.use_phoneme = "Pn" in name
        self.use_freq = "Fq" in name
        self.use_meg = "Meg" in name
        self.use_mri = "Mri" in name
        print(f"use_word: {self.use_word}, use_phoneme: {self.use_phoneme}, use_freq: {self.use_freq}, use_meg: {self.use_meg}, use_mri: {self.use_mri}")

        # where to load each file
        self.file_path_dict = create_file_path_dict(
            MEG_SUBJECT=MEG_SUBJECT,
            FMRI_SUBJECT=FMRI_SUBJECT,
            all_stories=self.all_stories,
            ds_freq=ds_freq,
            spacing=spacing,
            **kwargs,
        )

        # compute or accept precomputed PCAs
        if self.use_meg:
            self.pca_meg = pca_meg or self._pca_on_meg()
        else:
            self.pca_meg = None

        if self.use_mri:
            self.pca_mri = pca_mri or self._pca_on_mri()
        else:
            self.pca_mri = None
        
        # figure out embed_dim
        self.embed_dim = 0
        if self.use_word:
            self.embed_dim += 768
        if self.use_phoneme:
            self.embed_dim += 44
        if self.use_freq:
            self.embed_dim += 40
        if self.use_meg:
            self.embed_dim += self.pca_meg.n_components_
        if self.use_mri:
            self.embed_dim += self.pca_mri.n_components_
        print("embed_dim: ", self.embed_dim)

        # segment settings (must align to TR)
        assert seg_sec % tr == 0, "seg_sec must be multiple of tr"
        assert overlap_sec % tr == 0, "overlap_sec must be multiple of tr"
        self.seg_tr_len = seg_sec // tr  # #TR per segment
        self.step_tr = (seg_sec - overlap_sec) // tr
        self.seg_embed_len = seg_sec * self.ds_freq  # #samples per segment
        self.step_embed = (seg_sec - overlap_sec) * self.ds_freq

        # how many TRs we padded for fMRI-embed alignment
        self._pad_tr = 5

        # now preload & index all segments
        self._preload()
        print(f"Preloaded {len(self._segments)} segments.")

    def _preload(self):
        # load & process each story once
        self.story_data = {}
        for story in self.all_stories:
            embeds, meg, fmri = self._load_story_data(story)
            self.story_data[story] = {"embeds": embeds, "meg": meg, "fmri": fmri}

        # build a flat list of (story, tr_start) for every 40s-window
        self._segments = []
        for story, data in self.story_data.items():
            n_tr = data["fmri"].shape[0]
            n_segs = (n_tr - self.seg_tr_len) // self.step_tr + 1
            for j in range(n_segs):
                self._segments.append((story, j * self.step_tr))

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx):
        story, tr_start = self._segments[idx]
        data = self.story_data[story]

        # compute embed/meg indices
        start_embed = (tr_start + self._pad_tr) * self.tr * self.ds_freq
        end_embed = start_embed + self.seg_embed_len

        # get the segments
        embeds_seg = data["embeds"][start_embed:end_embed]
        meg_seg = data["meg"][start_embed:end_embed]
        fmri_seg = data["fmri"][tr_start : tr_start + self.seg_tr_len]

        return embeds_seg, meg_seg, fmri_seg

    def _load_story_data(self, story):
        """Load design, meg, fmri for a single story and apply PCA + alignment."""
        # 1) design mats
        mats = []
        if self.use_word:
            with open(self.file_path_dict["word"], "rb") as f:
                mats.append(pickle.load(f)[story])
        if self.use_phoneme:
            with open(self.file_path_dict["phoneme"], "rb") as f:
                mats.append(pickle.load(f)[story])
        if self.use_freq:
            with open(self.file_path_dict["spectrum"], "rb") as f:
                mats.append(pickle.load(f)[story])

        # trim to shortest and concat
        min_len = min(m.shape[0] for m in mats)
        mats = [m[:min_len] for m in mats]
        embeds = torch.from_numpy(np.concatenate(mats, axis=1)).float()

        # 2) MEG
        meg = mne.io.read_raw_fif(self.file_path_dict["meg"][story], verbose=False).pick("meg").load_data(verbose=False).get_data().T
        meg = torch.from_numpy(meg).float()
        meg = zscore_tensor(meg, dim=0)

        # 3) fMRI
        stc_fmri = mne.read_source_estimate(self.file_path_dict["fmri"][story])
        fmri = stc_fmri.data.T
        fmri = torch.from_numpy(fmri).float()
        fmri = zscore_tensor(fmri, dim=0)

        # 4) align lengths to fMRI
        n_tr = fmri.shape[0]
        T = (n_tr + self._pad_tr) * self.tr * self.ds_freq
        while embeds.shape[0] < T or meg.shape[0] < T:
            n_tr -= 1
            T = (n_tr + self._pad_tr) * self.tr * self.ds_freq
        fmri = fmri[:n_tr]
        embeds = embeds[:T]
        meg = meg[:T]

        # 5) add MEG-PCA features
        if self.use_meg:
            meg_pc = self.pca_meg.transform(meg.numpy())
            meg_pc = torch.from_numpy(meg_pc).float()
            embeds = torch.cat([embeds, meg_pc], dim=1)

        # 6) add fMRI-PCA features into embed
        if self.use_mri:  # pad 5 TR of zeros, then upsample by ds_freq
            fmri_pc = self.pca_mri.transform(fmri.numpy())
            fmri_pc = torch.from_numpy(fmri_pc).float()
            fmri_pc = torch.cat([torch.zeros(5, fmri_pc.shape[1]), fmri_pc], dim=0)
            fmri_pc = fmri_pc.repeat_interleave(self.ds_freq * self.tr, dim=0)
            embeds = torch.cat([embeds, fmri_pc], dim=1)

        return embeds, meg, fmri

    def _pca_on_meg(self, n_components=0.99):
        all_meg = []
        for story in self.all_stories:
            meg = mne.io.read_raw_fif(self.file_path_dict["meg"][story], verbose=False).pick("meg").load_data(verbose=False).get_data().T
            meg = nan_zscore(meg, axis=0)
            all_meg.append(meg)
        all_meg = np.vstack(all_meg)
        print(f"meg for PCA shape: {all_meg.shape}")
        pca = PCA(n_components=n_components).fit(all_meg)
        print("n components: ", pca.n_components_)
        print("meg pca explained variance ratio: ", np.sum(pca.explained_variance_ratio_))
        return pca

    def _pca_on_mri(self, n_components=0.99):
        all_fmri = []
        for story in self.all_stories:
            stc_fmri = mne.read_source_estimate(self.file_path_dict["fmri"][story])
            fmri = stc_fmri.data.T
            fmri = nan_zscore(fmri, axis=0)
            all_fmri.append(fmri)
        all_fmri = np.vstack(all_fmri)
        print(f"fmri for PCA shape: {all_fmri.shape}")
        pca = PCA(n_components=n_components).fit(all_fmri)
        print("n components: ", pca.n_components_)
        print("fmri pca explained variance ratio: ", np.sum(pca.explained_variance_ratio_))
        return pca
        
        
