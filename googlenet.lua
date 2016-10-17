require 'nn'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

function InceptionModule(name, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5, outplane_pool_proj)
  local a = nn.Sequential()
  local a1x1 = nn.SpatialConvolution(inplane, outplane_a1x1, 1, 1, 1, 1, 0, 0)
  a1x1.name = name .. '_1x1'
  a:add(a1x1)
  a:add(nn.ReLU(true))

  local b = nn.Sequential()
  local b3x3_reduce = nn.SpatialConvolution(inplane, outplane_b3x3_reduce, 1, 1, 1, 1, 0, 0)
  b3x3_reduce.name = name .. '_3x3_reduce'
  b:add(b3x3_reduce)
  b:add(nn.ReLU(true))
  local b3x3 = nn.SpatialConvolution(outplane_b3x3_reduce, outplane_b3x3, 3, 3, 1, 1, 1, 1)
  b3x3.name = name .. '_3x3'
  b:add(b3x3)
  b:add(nn.ReLU(true))

  local c = nn.Sequential()
  local c5x5_reduce = nn.SpatialConvolution(inplane, outplane_c5x5_reduce, 1, 1, 1, 1, 0, 0)
  c5x5_reduce.name = name .. '_5x5_reduce'
  c:add(c5x5_reduce)
  c:add(nn.ReLU(true))
  local c5x5 = nn.SpatialConvolution(outplane_c5x5_reduce, outplane_c5x5, 5, 5, 1, 1, 2, 2)
  c5x5.name = name .. '_5x5'
  c:add(c5x5)
  c:add(nn.ReLU(true))

  local d = nn.Sequential()
  d:add(nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1))
  local d_pool_proj = nn.SpatialConvolution(inplane, outplane_pool_proj, 1, 1, 1, 1, 0, 0)
  d_pool_proj.name = name .. '_pool_proj'
  d:add(d_pool_proj)
  d:add(nn.ReLU(true))

  local module = nn.Sequential():add(nn.Concat(2):add(a):add(b):add(c):add(d))
  return module
end

local model = nn.Sequential()

local conv1_7x7_s2 = nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)
conv1_7x7_s2.name = 'conv1_7x7_s2'
model:add(conv1_7x7_s2)
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
model:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75, 1))

local conv_3x3_reduce = nn.SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0)
conv_3x3_reduce.name = 'conv2_3x3_reduce'
model:add(conv_3x3_reduce)
model:add(nn.ReLU(true))
local conv_3x3 = nn.SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
conv_3x3.name = 'conv2_3x3'
model:add(conv_3x3)
model:add(nn.ReLU(true))
model:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75, 1))
model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())

local bf_inception = model:clone()

model:add(InceptionModule('inception_3a', 192, 64, 96, 128, 16, 32, 32))
model:add(InceptionModule('inception_3b', 256, 128, 128, 192, 32, 96, 64))

model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())

model:add(InceptionModule('inception_4a', 480, 192, 96, 208, 16, 48, 64))
local bf_sftmax1 = model:clone()

model:add(InceptionModule('inception_4b', 512, 160, 112, 224, 24, 64, 64))
model:add(InceptionModule('inception_4c', 512, 128, 128, 256, 24, 64, 64))
model:add(InceptionModule('inception_4d', 512, 112, 144, 288, 32, 64, 64))
local bf_sftmax2 = model:clone()

model:add(InceptionModule('inception_4e', 528, 256, 160, 320, 32, 128, 128))

model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())

model:add(InceptionModule('inception_5a', 832, 256, 160, 320, 32, 128, 128))
model:add(InceptionModule('inception_5b', 832, 384, 192, 384, 48, 128, 128))


------------------------------------------------------------------
--*************   softmax3(exclude softmax layer   *************--
------------------------------------------------------------------
model:add(nn.SpatialAveragePooling(7, 7, 1, 1))
model:add(nn.View(-1, 1024))

--model:add(nn.Dropout(0.4))

local classifier3 = nn.Linear(1024, 1000)
classifier3.name = 'loss3_classifier'
model:add(classifier3)

-----------------------------------------------------------------


------------------------------------------------------------------
--*************   af_softmax1(exclude softmax layer   *************--
------------------------------------------------------------------
local af_softmax1 = nn.Sequential()
af_softmax1:add(nn.SpatialAveragePooling(5, 5, 3, 3))

local conv1_1x1_s1 = nn.SpatialConvolution(512, 128, 1, 1)
conv1_1x1_s1.name = 'loss1_conv'
af_softmax1:add(conv1_1x1_s1)
af_softmax1:add(nn.ReLU(true))
af_softmax1:add(nn.View(128*4*4):setNumInputDims(3))

local linear_sftmax1 = nn.Linear(128*4*4, 1024)
linear_sftmax1.name = 'loss1_fc'
af_softmax1:add(linear_sftmax1)
af_softmax1:add(nn.ReLU(true))
--bf_sftmax1:add(nn.Dropout(0.7))

local classifier1 = nn.Linear(1024, 1000)
classifier1.name = 'loss1_classifier'
af_softmax1:add(classifier1)
-------------------------------------------------------------------



------------------------------------------------------------------
--*************   af_softmax2(exclude softmax layer   *************--
------------------------------------------------------------------
local af_softmax2 = nn.Sequential()
af_softmax2:add(nn.SpatialAveragePooling(5, 5, 3, 3))

local conv1_1x1_s1_copy = nn.SpatialConvolution(528, 128, 1, 1)
conv1_1x1_s1_copy.name = 'loss2_conv'
af_softmax2:add(conv1_1x1_s1_copy)
af_softmax2:add(nn.ReLU(true))

af_softmax2:add(nn.View(128*4*4):setNumInputDims(3))

local linear_sftmax2 = nn.Linear(128*4*4, 1024)
linear_sftmax2.name = 'loss2_fc'
af_softmax2:add(linear_sftmax2)
af_softmax2:add(nn.ReLU(true))
--bf_sftmax2:add(nn.Dropout(0.7))

local classifier2 = nn.Linear(1024, 1000)
classifier2.name = 'loss2_classifier'
af_softmax2:add(classifier2)
------------------------------------------------------------------


local paramsFile = hdf5.open('train_val.hdf5', 'r')
--local moduleQueue = {  af_softmax1, af_softmax2}
local moduleQueue = { model, af_softmax1, af_softmax2 }
local touchedLayers = { }
local inputData = paramsFile:read('InputLayer'):all()


for k1, v1 in ipairs(moduleQueue) do
  if v1.modules then
    for k2, v2 in ipairs(v1.modules) do
      table.insert(moduleQueue, v2)
    end
  end

  if v1.name then
     touchedLayers[v1.name] = true
     weight_name = v1.name .. "___" .. "weight"

     print(weight_name)
     
     local layer_weight = paramsFile:read(weight_name):all()
     if layer_weight then
         v1.weight:copy(layer_weight)
     
         bias_name = v1.name .. "___" .. "bias"
         print(bias_name)
         local layer_bias = paramsFile:read(bias_name):all()
         if layer_bias then
           v1.bias:copy(layer_bias)
         else
           print(v1.name .. ' has no bias')
         end
     end
   end
end

paramsFile:close()
print('  ')
print('  ')
print(inputData:size())
print('  ')
print('  ')
print('before inception')
bf_inception = bf_inception:float()
--print(bf_inception)
torch.save('googlenet_bf_inception.t7', bf_inception)

print('  ')
print('softmax3')
softmax3 = model:float()
--print(softmax3)
torch.save('googlenet_softmax3.t7', softmax3)


print('  ')
print('softmax1')
softmax1 = bf_sftmax1:add(af_softmax1):float()
--print(softmax1)
torch.save('googlenet_softmax1.t7', softmax1)


print('  ')
print('softmax2')
softmax2 = bf_sftmax2:add(af_softmax2):float()
--print(softmax2)
torch.save('googlenet_softmax2.t7', softmax2)




