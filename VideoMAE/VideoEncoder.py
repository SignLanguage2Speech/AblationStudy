from transformers import AutoImageProcessor, VideoMAEModel, VideoMAEForVideoClassification
import torch.nn as nn
import torch

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        
        headModel = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # self.conv_head = nn.Sequential(
        #     nn.Conv3d(1, 1, (3,5,5), stride = (1,3,3), padding = (1,0,0)),
        #     nn.MaxPool3d((1,2,2))
        # )

        self.linear_head = nn.Sequential(
            headModel.fc_norm,
            nn.Linear(in_features=768, out_features=1024, bias=True),
        )

    def batch_videos(self, videos, lens):
        num_videos = -(videos[0].size(0) // -16)
        num_videos_all = num_videos*videos.size(0)
        bathced_videos = torch.zeros(num_videos_all,16,3,224,224, device = self.device)

        current_video_index = 0
        for n in range(len(videos)):
            for i in range(num_videos):
                video_start = i*16
                video_end = min((i+1)*16, len(videos[n]))
                batched_video_len = video_end-video_start
                bathced_videos[current_video_index,:batched_video_len] = videos[n,video_start:video_end]
                current_video_index += 1
        
        num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
        seq_length = (16 // self.model.config.tubelet_size) * num_patches_per_frame
        bool_masked_pos = torch.zeros(num_videos_all, seq_length).bool()

        inputs = {
            'pixel_values': bathced_videos, 
            # 'bool_masked_pos': bool_masked_pos,
            }
        return inputs, num_videos

    def unbatch_videos(self, x, num_videos, lens):
        unbatched_videos = torch.zeros(len(lens),num_videos, 1024, device = self.device)
        for i in range(len(lens)):
            unbatched_videos[i] = x[i*num_videos: (i+1)*num_videos]
        return unbatched_videos

    def forward(self, x, lens):
        x = x.permute(0,2,1,3,4)
        inputs, num_videos = self.batch_videos(x, lens)
        x = self.model(**inputs).last_hidden_state
        x = self.linear_head(x.mean(1))
        x = self.unbatch_videos(x, num_videos, lens)
        # x = x.unsqueeze(1)
        # x = self.conv_head(x)
        # x = x.squeeze(1)
        # x = x.flatten(start_dim = 2)
        # print(x.size())
        return x
    



# CFG = {'model_path': "MCG-NJU/videomae-base"}
# VideoEncoder = VideoEncoder(CFG)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# videos = torch.rand(4,29,3,224,224, device = device)
# lens = [29,20,20,20]

# VideoEncoder.forward(videos, lens)