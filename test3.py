c1, c2 = (int(box[0].item() if isinstance(box[0], torch.Tensor) else box[0]), 
              int(box[1].item() if isinstance(box[1], torch.Tensor) else box[1])), \
             (int(box[2].item() if isinstance(box[2], torch.Tensor) else box[2]), 
              int(box[3].item() if isinstance(box[3], torch.Tensor) else box[3]))