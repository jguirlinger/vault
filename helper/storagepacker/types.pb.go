// Code generated by protoc-gen-go. DO NOT EDIT.
// source: types.proto

package storagepacker

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"
import any "github.com/golang/protobuf/ptypes/any"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion2 // please upgrade the proto package

// Item represents a entry that gets inserted into the storage packer
type Item struct {
	// ID is the UUID to identify the item
	ID string `sentinel:"" protobuf:"bytes,1,opt,name=id" json:"id,omitempty"`
	// message is the contents of the item
	Message              *any.Any `sentinel:"" protobuf:"bytes,2,opt,name=message" json:"message,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Item) Reset()         { *m = Item{} }
func (m *Item) String() string { return proto.CompactTextString(m) }
func (*Item) ProtoMessage()    {}
func (*Item) Descriptor() ([]byte, []int) {
	return fileDescriptor_types_cb4c4b01a6229b9f, []int{0}
}
func (m *Item) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Item.Unmarshal(m, b)
}
func (m *Item) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Item.Marshal(b, m, deterministic)
}
func (dst *Item) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Item.Merge(dst, src)
}
func (m *Item) XXX_Size() int {
	return xxx_messageInfo_Item.Size(m)
}
func (m *Item) XXX_DiscardUnknown() {
	xxx_messageInfo_Item.DiscardUnknown(m)
}

var xxx_messageInfo_Item proto.InternalMessageInfo

func (m *Item) GetID() string {
	if m != nil {
		return m.ID
	}
	return ""
}

func (m *Item) GetMessage() *any.Any {
	if m != nil {
		return m.Message
	}
	return nil
}

// BucketV2 is a construct to hold multiple items within itself. This
// abstraction contains multiple buckets of the same kind within itself and
// shares amont them the items that get inserted. When the bucket as a whole
// gets too big to hold more items, the contained buckets gets pushed out only
// to become independent buckets. Hence, this can grow infinitely in terms of
// storage space for items that get inserted.
type BucketV2 struct {
	// Key is the storage path where the bucket gets stored
	Key string `sentinel:"" protobuf:"bytes,1,opt,name=key" json:"key,omitempty"`
	// Buckets are the buckets contained within this bucket
	Buckets map[string]*BucketV2 `sentinel:"" protobuf:"bytes,2,rep,name=buckets" json:"buckets,omitempty" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	// Items holds the items contained within this bucket
	Items map[string]*Item `sentinel:"" protobuf:"bytes,3,rep,name=items" json:"items,omitempty" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	// Sharded indicates if the buckets contained in this bucket are pushed out
	// or not
	Sharded bool `sentinel:"" protobuf:"varint,4,opt,name=sharded" json:"sharded,omitempty"`
	// Depth indicates the hierarchical positioning of this bucket
	Depth int32 `sentinel:"" protobuf:"varint,5,opt,name=depth" json:"depth,omitempty"`
	// Size of this bucket in number of bytes
	Size                 int64    `sentinel:"" protobuf:"varint,6,opt,name=size" json:"size,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *BucketV2) Reset()         { *m = BucketV2{} }
func (m *BucketV2) String() string { return proto.CompactTextString(m) }
func (*BucketV2) ProtoMessage()    {}
func (*BucketV2) Descriptor() ([]byte, []int) {
	return fileDescriptor_types_cb4c4b01a6229b9f, []int{1}
}
func (m *BucketV2) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BucketV2.Unmarshal(m, b)
}
func (m *BucketV2) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BucketV2.Marshal(b, m, deterministic)
}
func (dst *BucketV2) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BucketV2.Merge(dst, src)
}
func (m *BucketV2) XXX_Size() int {
	return xxx_messageInfo_BucketV2.Size(m)
}
func (m *BucketV2) XXX_DiscardUnknown() {
	xxx_messageInfo_BucketV2.DiscardUnknown(m)
}

var xxx_messageInfo_BucketV2 proto.InternalMessageInfo

func (m *BucketV2) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *BucketV2) GetBuckets() map[string]*BucketV2 {
	if m != nil {
		return m.Buckets
	}
	return nil
}

func (m *BucketV2) GetItems() map[string]*Item {
	if m != nil {
		return m.Items
	}
	return nil
}

func (m *BucketV2) GetSharded() bool {
	if m != nil {
		return m.Sharded
	}
	return false
}

func (m *BucketV2) GetDepth() int32 {
	if m != nil {
		return m.Depth
	}
	return 0
}

func (m *BucketV2) GetSize() int64 {
	if m != nil {
		return m.Size
	}
	return 0
}

type Bucket struct {
	Key                  string   `sentinel:"" protobuf:"bytes,1,opt,name=key" json:"key,omitempty"`
	Items                []*Item  `sentinel:"" protobuf:"bytes,2,rep,name=items" json:"items,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Bucket) Reset()         { *m = Bucket{} }
func (m *Bucket) String() string { return proto.CompactTextString(m) }
func (*Bucket) ProtoMessage()    {}
func (*Bucket) Descriptor() ([]byte, []int) {
	return fileDescriptor_types_cb4c4b01a6229b9f, []int{2}
}
func (m *Bucket) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Bucket.Unmarshal(m, b)
}
func (m *Bucket) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Bucket.Marshal(b, m, deterministic)
}
func (dst *Bucket) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Bucket.Merge(dst, src)
}
func (m *Bucket) XXX_Size() int {
	return xxx_messageInfo_Bucket.Size(m)
}
func (m *Bucket) XXX_DiscardUnknown() {
	xxx_messageInfo_Bucket.DiscardUnknown(m)
}

var xxx_messageInfo_Bucket proto.InternalMessageInfo

func (m *Bucket) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *Bucket) GetItems() []*Item {
	if m != nil {
		return m.Items
	}
	return nil
}

func init() {
	proto.RegisterType((*Item)(nil), "storagepacker.Item")
	proto.RegisterType((*BucketV2)(nil), "storagepacker.BucketV2")
	proto.RegisterMapType((map[string]*BucketV2)(nil), "storagepacker.BucketV2.BucketsEntry")
	proto.RegisterMapType((map[string]*Item)(nil), "storagepacker.BucketV2.ItemsEntry")
	proto.RegisterType((*Bucket)(nil), "storagepacker.Bucket")
}

func init() { proto.RegisterFile("types.proto", fileDescriptor_types_cb4c4b01a6229b9f) }

var fileDescriptor_types_cb4c4b01a6229b9f = []byte{
	// 317 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x91, 0x4f, 0x4b, 0x03, 0x31,
	0x10, 0xc5, 0xc9, 0x6e, 0xb7, 0xad, 0x53, 0x15, 0x89, 0x05, 0x63, 0x4f, 0xcb, 0xe2, 0x21, 0x1e,
	0x4c, 0xa1, 0x5e, 0x8a, 0x07, 0x41, 0xa1, 0x82, 0x07, 0x2f, 0x11, 0xbc, 0xa7, 0xdd, 0x71, 0xbb,
	0xf4, 0xcf, 0x2e, 0x9b, 0x54, 0x58, 0x3f, 0xbc, 0x48, 0x93, 0x06, 0x5b, 0xd9, 0xde, 0x66, 0x98,
	0xf7, 0x7e, 0x79, 0x33, 0x81, 0x9e, 0xa9, 0x4b, 0xd4, 0xa2, 0xac, 0x0a, 0x53, 0xd0, 0x33, 0x6d,
	0x8a, 0x4a, 0x65, 0x58, 0xaa, 0xd9, 0x02, 0xab, 0xc1, 0x75, 0x56, 0x14, 0xd9, 0x12, 0x87, 0x76,
	0x38, 0xdd, 0x7c, 0x0e, 0xd5, 0xba, 0x76, 0xca, 0xe4, 0x05, 0x5a, 0xaf, 0x06, 0x57, 0xf4, 0x1c,
	0x82, 0x3c, 0x65, 0x24, 0x26, 0xfc, 0x44, 0x06, 0x79, 0x4a, 0x05, 0x74, 0x56, 0xa8, 0xb5, 0xca,
	0x90, 0x05, 0x31, 0xe1, 0xbd, 0x51, 0x5f, 0x38, 0x88, 0xf0, 0x10, 0xf1, 0xb4, 0xae, 0xa5, 0x17,
	0x25, 0x3f, 0x01, 0x74, 0x9f, 0x37, 0xb3, 0x05, 0x9a, 0x8f, 0x11, 0xbd, 0x80, 0x70, 0x81, 0xf5,
	0x8e, 0xb6, 0x2d, 0xe9, 0x23, 0x74, 0xa6, 0x76, 0xaa, 0x59, 0x10, 0x87, 0xbc, 0x37, 0xba, 0x11,
	0x07, 0x11, 0x85, 0xf7, 0xee, 0x0a, 0x3d, 0x59, 0x9b, 0xaa, 0x96, 0xde, 0x44, 0xc7, 0x10, 0xe5,
	0x06, 0x57, 0x9a, 0x85, 0xd6, 0x9d, 0x1c, 0x73, 0x6f, 0x77, 0xd9, 0x79, 0x9d, 0x81, 0x32, 0xe8,
	0xe8, 0xb9, 0xaa, 0x52, 0x4c, 0x59, 0x2b, 0x26, 0xbc, 0x2b, 0x7d, 0x4b, 0xfb, 0x10, 0xa5, 0x58,
	0x9a, 0x39, 0x8b, 0x62, 0xc2, 0x23, 0xe9, 0x1a, 0x4a, 0xa1, 0xa5, 0xf3, 0x6f, 0x64, 0xed, 0x98,
	0xf0, 0x50, 0xda, 0x7a, 0xf0, 0x0e, 0xa7, 0xfb, 0xb1, 0x1a, 0xf6, 0xbb, 0x83, 0xe8, 0x4b, 0x2d,
	0x37, 0xfe, 0x58, 0x57, 0x47, 0xf2, 0x49, 0xa7, 0x7a, 0x08, 0xc6, 0x64, 0xf0, 0x06, 0xf0, 0x97,
	0xb6, 0x01, 0x79, 0x7b, 0x88, 0xbc, 0xfc, 0x87, 0xdc, 0x7a, 0xf7, 0x70, 0xc9, 0x04, 0xda, 0xee,
	0x95, 0x66, 0x94, 0xbb, 0x9e, 0xbb, 0x7d, 0x33, 0xca, 0x2a, 0xa6, 0x6d, 0xfb, 0xbd, 0xf7, 0xbf,
	0x01, 0x00, 0x00, 0xff, 0xff, 0x53, 0xc1, 0x10, 0x40, 0x4f, 0x02, 0x00, 0x00,
}
