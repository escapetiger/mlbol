import mlbol as mb

# rect = mb.geometry.HyperRectangle([0., 0.], [1., 1.])
# x, w = rect.quadrature_points(16, mode='gausslegd')
# print(x)
# print(w)

# D1 = mb.data.SequenceTensorDataset(mb.rand((10, 3)))
# D2 = mb.data.SequenceTensorDataset(mb.rand((5, 3)))
# D3 = mb.data.MappingTensorDataset(x=mb.rand((15, 3)), y=mb.rand((15, 3)))
# D4 = mb.data.MappingTensorDataset(x=mb.rand((25, 3)), y=mb.rand((25, 3)))
# # Example data
# data = {"pde": D1, "bc": D2}
# dataloader = {
#     "pde": mb.data.BatchLoader(D1, batch_size=4, drop_last=False, shuffle=False),
#     "bc": mb.data.BatchLoader(D2, batch_size=2),
# }
# datazip = mb.data.TreeLoader(dataloader, mode="max_size_cycle")
# for batch, batch_idx, _ in datazip:
#     print(
#         f"Batch: {batch}, Batch Index: {batch_idx}, Iterable Index: {_}"
#     )

dom = mb.geometry.Domain(
    {
        "t": mb.geometry.TimeInterval(1),
        "x": mb.geometry.Interval(0, 1),
        "v": mb.geometry.Interval(-1, 1),
    }
)

# x = dom.boundary_points('t', 4, [2, 2], "uniform", "random", {}, {"mode": "pseudo"})
# print(dom['t'])
# print(dom['x'])
# print(dom['v'])
# print(x)
# print(x.shape)
