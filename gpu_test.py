import torch


device = "cpu"

test_var = torch.eye(100).to(device)

test_va2 = torch.eye(100).to(device)

torch.matmul(test_var, test_va2)

print("Job Executed Successfully")
