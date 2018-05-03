package storagepacker

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	radix "github.com/armon/go-radix"
	"github.com/golang/protobuf/proto"
	"github.com/hashicorp/errwrap"
	"github.com/hashicorp/vault/helper/strutil"

	log "github.com/hashicorp/go-hclog"
	"github.com/hashicorp/vault/helper/cryptoutil"
	"github.com/hashicorp/vault/logical"
)

const (
	defaultBucketBaseCount  = 256
	defaultBucketShardCount = 16
	// Larger size of the bucket size adversely affects the performance of the
	// storage packer. Also, some of the backends impose a maximum size limit
	// on the objects that gets persisted. For example, Consul imposes 512KB
	// and DynamoDB imposes 400KB. Going forward, if there exists storage
	// backends that has more constrained limits, this will have to become more
	// flexible. For now, 380KB seems like a decent bargain.
	defaultBucketMaxSize = 380 * 1024
)

type Config struct {
	// View is the storage to be used by all the buckets
	View logical.Storage

	// ViewPrefix is the prefix to be used for the buckets in the view
	ViewPrefix string

	// Logger for output
	Logger log.Logger

	// BucketBaseCount is the number of buckets to create at the base level
	BucketBaseCount int

	// BucketShardCount is the number of sub-buckets a bucket gets sharded into
	// when it reaches the maximum threshold
	BucketShardCount int

	// BucketMaxSize (in bytes) is the maximum allowed size per bucket. When
	// the size of the bucket reaches a threshold relative to this limit, it
	// gets sharded into the configured number of pieces incrementally.
	BucketMaxSize int64
}

// StoragePackerV2 packs many items into abstractions called buckets. The goal
// is to employ a reduced number of storage entries for a relatively huge
// number of items. This is the second version of the utility which supports
// indefinitely expanding the capacity of the storage by sharding the buckets
// when they exceed the imposed limit.
type StoragePackerV2 struct {
	config       *Config
	bucketsCache *radix.Tree
}

type LockedBucket struct {
	sync.RWMutex
	Data *BucketV2
}

// Clone creates a replica of the bucket
func (b *BucketV2) Clone() (*BucketV2, error) {
	if b == nil {
		return nil, fmt.Errorf("nil bucket")
	}

	marshaledBucket, err := proto.Marshal(b)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal bucket: %v", err)
	}

	var clonedBucket BucketV2
	err = proto.Unmarshal(marshaledBucket, &clonedBucket)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal bucket: %v", err)
	}

	return &clonedBucket, nil
}

// putItem is a recursive function that finds the appropriate bucket
// to store the item based on the storage space available in the buckets.
func (s *StoragePackerV2) putItem(bucket *LockedBucket, item *Item) (string, error) {
	// Bucket will be nil for the first time when its not known which base
	// level bucket the item belongs to.
	if bucket == nil {
		// Compute the index of the base bucket
		baseIndex, err := s.baseBucketIndex(item.ID)
		if err != nil {
			return "", err
		}

		// Prepend the index with the prefix
		baseKey := s.config.ViewPrefix + baseIndex

		// Check if the base bucket exists
		bucket, err = s.GetBucket(baseKey)
		if err != nil {
			return "", err
		}

		// If the base bucket does not exist, create one
		if bucket == nil {
			bucket = &LockedBucket{
				Data: s.newBucket(baseKey, 0),
			}
		}
	}

	// For sanity
	if bucket == nil {
		return "", fmt.Errorf("bucket is nil")
	}

	// Compute the shard index to which the item belongs
	shardIndex, err := shardBucketIndex(item.ID, int(bucket.Data.Depth), int(s.config.BucketBaseCount), int(s.config.BucketShardCount))
	if err != nil {
		return "", errwrap.Wrapf("failed to compute the bucket shard index: {{err}}", err)
	}
	shardKey := bucket.Data.Key + "/" + shardIndex

	bucket.Lock()

	if bucket.Data.Sharded {
		bucket.Unlock()
		bucketShard, err := s.GetBucket(shardKey)
		if err != nil {
			return "", err
		}
		return s.putItem(bucketShard, item)
	}

	defer bucket.Unlock()

	bucketShard, ok := bucket.Data.Buckets[shardIndex]
	if !ok {
		bucketShard = s.newBucket(shardKey, bucket.Data.Depth+1)
		bucket.Data.Buckets[shardIndex] = bucketShard
	}

	if bucketShard == nil {
		bucket.Unlock()
		return "", fmt.Errorf("bucket shard is nil")
	}

	limitExceeded, err := s.bucketExceedsSizeLimit(bucket, item)
	if err != nil {
		return "", err
	}

	// If the bucket size is within the limit, return the updated bucket
	if !limitExceeded {
		bucketShard.Items[item.ID] = item
		return bucketShard.Key, s.PutBucket(bucket)
	}

	err = s.splitBucket(bucket)
	if err != nil {
		return "", err
	}

	shardedBucket, err := s.GetBucket(bucketShard.Key)
	if err != nil {
		return "", err
	}

	bucketKey, err := s.putItem(shardedBucket, item)
	if err != nil {
		return "", err
	}

	return bucketKey, s.PutBucket(bucket)
}

// Get reads a bucket from the storage
func (s *StoragePackerV2) GetBucket(key string) (*LockedBucket, error) {
	if key == "" {
		return nil, fmt.Errorf("missing bucket key")
	}

	raw, exists := s.bucketsCache.Get(key)
	if exists {
		return raw.(*LockedBucket), nil
	}

	// Read from the underlying view
	entry, err := s.config.View.Get(context.Background(), key)
	if err != nil {
		return nil, errwrap.Wrapf("failed to read bucket: {{err}}", err)
	}
	if entry == nil {
		return nil, nil
	}

	var bucket BucketV2
	err = proto.Unmarshal(entry.Value, &bucket)
	if err != nil {
		return nil, errwrap.Wrapf("failed to decode bucket: {{err}}", err)
	}

	// Serializing and deserializing a proto message with empty map translates
	// to a nil. Ensure that the required fields are initialized properly.
	if bucket.Buckets == nil {
		bucket.Buckets = make(map[string]*BucketV2)
	}
	if bucket.Items == nil {
		bucket.Items = make(map[string]*Item)
	}

	bucket.Size = int64(len(entry.Value))

	return s.UpdateBucketCache(&LockedBucket{Data: &bucket}), nil
}

func (s *StoragePackerV2) UpdateBucketCache(bucket *LockedBucket) *LockedBucket {
	var lb *LockedBucket
	raw, exists := s.bucketsCache.Get(bucket.Data.Key)
	if exists {
		lb = raw.(*LockedBucket)
	}
	if lb == nil {
		lb = bucket
	}
	s.bucketsCache.Insert(lb.Data.Key, lb)
	return lb
}

// Put stores a bucket in storage
func (s *StoragePackerV2) PutBucket(bucket *LockedBucket) error {
	if bucket == nil {
		return fmt.Errorf("nil bucket entry")
	}

	if bucket.Data.Key == "" {
		return fmt.Errorf("missing bucket key")
	}

	if !strings.HasPrefix(bucket.Data.Key, s.config.ViewPrefix) {
		return fmt.Errorf("bucket entry key should have %q prefix", s.config.ViewPrefix)
	}

	marshaledBucket, err := proto.Marshal(bucket.Data)
	if err != nil {
		return err
	}

	err = s.config.View.Put(context.Background(), &logical.StorageEntry{
		Key:   bucket.Data.Key,
		Value: marshaledBucket,
	})
	if err != nil {
		return err
	}

	bucket.Data.Size = int64(len(marshaledBucket))

	s.UpdateBucketCache(bucket)

	return nil
}

// getItem is a recursive function that fetches the given item ID in
// the bucket hierarchy
func (s *StoragePackerV2) getItem(bucket *LockedBucket, itemID string) (*Item, error) {
	if bucket == nil {
		baseIndex, err := s.baseBucketIndex(itemID)
		if err != nil {
			return nil, err
		}

		bucket, err = s.GetBucket(s.config.ViewPrefix + baseIndex)
		if err != nil {
			return nil, errwrap.Wrapf("failed to read packed storage item: {{err}}", err)
		}
	}

	if bucket == nil {
		return nil, nil
	}

	shardIndex, err := shardBucketIndex(itemID, int(bucket.Data.Depth), int(s.config.BucketBaseCount), int(s.config.BucketShardCount))
	if err != nil {
		return nil, errwrap.Wrapf("failed to compute the bucket shard index: {{err}}", err)
	}

	shardKey := bucket.Data.Key + "/" + shardIndex

	bucket.RLock()

	if bucket.Data.Sharded {
		bucket.RUnlock()
		bucketShard, err := s.GetBucket(shardKey)
		if err != nil {
			return nil, err
		}
		return s.getItem(bucketShard, itemID)
	}

	defer bucket.RUnlock()

	bucketShard, ok := bucket.Data.Buckets[shardIndex]
	if !ok {
		return nil, nil
	}

	return bucketShard.Items[itemID], nil
}

// deleteItem is a recursive function that finds the bucket holding
// the item and removes the item from it
func (s *StoragePackerV2) deleteItem(bucket *LockedBucket, itemID string) error {
	if bucket == nil {
		baseIndex, err := s.baseBucketIndex(itemID)
		if err != nil {
			return err
		}

		bucket, err = s.GetBucket(s.config.ViewPrefix + baseIndex)
		if err != nil {
			return errwrap.Wrapf("failed to read packed storage item: {{err}}", err)
		}
	}

	if bucket == nil {
		return nil
	}

	shardIndex, err := shardBucketIndex(itemID, int(bucket.Data.Depth), int(s.config.BucketBaseCount), int(s.config.BucketShardCount))
	if err != nil {
		return errwrap.Wrapf("failed to compute the bucket shard index: {{err}}", err)
	}

	shardKey := bucket.Data.Key + "/" + shardIndex

	bucket.Lock()

	if bucket.Data.Sharded {
		bucket.Unlock()
		bucketShard, err := s.GetBucket(shardKey)
		if err != nil {
			return err
		}
		return s.deleteItem(bucketShard, itemID)
	}

	defer bucket.Unlock()

	bucketShard, ok := bucket.Data.Buckets[shardIndex]
	if !ok {
		return nil
	}

	delete(bucketShard.Items, itemID)

	return s.PutBucket(bucket)
}

// GetItem fetches the item using the given item identifier
func (s *StoragePackerV2) GetItem(itemID string) (*Item, error) {
	if itemID == "" {
		return nil, fmt.Errorf("empty item ID")
	}

	return s.getItem(nil, itemID)
}

// PutItem persists the given item
func (s *StoragePackerV2) PutItem(item *Item) (string, error) {
	if item == nil {
		return "", fmt.Errorf("nil item")
	}

	if item.ID == "" {
		return "", fmt.Errorf("missing ID in item")
	}

	return s.putItem(nil, item)
}

// DeleteItem removes the item using the given item identifier
func (s *StoragePackerV2) DeleteItem(itemID string) error {
	if itemID == "" {
		return fmt.Errorf("empty item ID")
	}

	return s.deleteItem(nil, itemID)
}

// bucketExceedsSizeLimit indicates if the given bucket is exceeding the
// configured size limit on the storage packer
func (s *StoragePackerV2) bucketExceedsSizeLimit(bucket *LockedBucket, item *Item) (bool, error) {
	marshaledItem, err := proto.Marshal(item)
	if err != nil {
		return false, fmt.Errorf("failed to marshal item: %v", err)
	}

	size := bucket.Data.Size + int64(len(marshaledItem))

	// The objects that leave storage packer to get persisted get inflated due
	// to extra bits coming off of encryption. So, we consider the bucket full
	// much earlier to compensate the overhead. Testing with considering the
	// 70% of the max size as the limit resulted in object sizes dangerously
	// close to the actual limit. Hence, setting 60% as the cut-off value.
	max := math.Ceil((float64(s.config.BucketMaxSize) * float64(60)) / float64(100))

	return float64(size) > max, nil
}

type BucketWalkFunc func(item *Item) error

func (s *StoragePackerV2) BucketWalk(key string, fn BucketWalkFunc) error {
	bucket, err := s.GetBucket(key)
	if err != nil {
		return err
	}
	if bucket == nil {
		return nil
	}

	if !bucket.Data.Sharded {
		for _, bucket := range bucket.Data.Buckets {
			for _, item := range bucket.Items {
				err := fn(item)
				if err != nil {
					return err
				}
			}
		}
		return nil
	}

	for i := 0; i < s.config.BucketShardCount; i++ {
		shardKey := bucket.Data.Key + "/" + strconv.FormatInt(int64(i), 16)
		err = s.BucketWalk(shardKey, fn)
		if err != nil {
			return err
		}
	}

	return nil
}

func (s *StoragePackerV2) splitBucket(bucket *LockedBucket) error {
	for _, shard := range bucket.Data.Buckets {
		for itemID, item := range shard.Items {
			if shard.Buckets == nil {
				shard.Buckets = make(map[string]*BucketV2)
			}
			subShardIndex, err := shardBucketIndex(itemID, int(shard.Depth), int(s.config.BucketBaseCount), int(s.config.BucketShardCount))
			if err != nil {
				return err
			}
			subShard, ok := shard.Buckets[subShardIndex]
			if !ok {
				subShardKey := shard.Key + "/" + subShardIndex
				subShard = s.newBucket(subShardKey, shard.Depth+1)
				shard.Buckets[subShardIndex] = subShard
			}
			subShard.Items[itemID] = item
		}

		shard.Items = nil
		err := s.PutBucket(&LockedBucket{Data: shard})
		if err != nil {
			return err
		}
	}
	bucket.Data.Buckets = nil
	bucket.Data.Sharded = true
	return nil
}

// baseBucketIndex returns the index of the base bucket to which the
// given item belongs
func (s *StoragePackerV2) baseBucketIndex(itemID string) (string, error) {
	// Hash the item ID
	hashVal, err := cryptoutil.Blake2b256Hash(itemID)
	if err != nil {
		return "", err
	}

	// Extract the index value of the base bucket from the hash of the item ID
	return strutil.BitMaskedIndexHex(hashVal, bitsNeeded(s.config.BucketBaseCount))
}

// shardBucketIndex returns the index of the bucket shard to which the given
// item belongs at a particular depth.
func shardBucketIndex(itemID string, depth, bucketBaseCount, bucketShardCount int) (string, error) {
	// Hash the item ID
	hashVal, err := cryptoutil.Blake2b256Hash(itemID)
	if err != nil {
		return "", err
	}

	// Compute the bits required to enumerate base buckets
	shardsBitCount := bitsNeeded(bucketShardCount)

	// Compute the bits that are already consumed by the base bucket and the
	// shards at previous levels.
	ignoreBits := bitsNeeded(bucketBaseCount) + depth*shardsBitCount

	// Extract the index value of the bucket shard from the hash of the item ID
	return strutil.BitMaskedIndexHex(hashVal[ignoreBits:], shardsBitCount)
}

// bitsNeeded returns the minimum number of bits required to enumerate the
// natural numbers below the given value
func bitsNeeded(value int) int {
	return int(math.Ceil(math.Log2(float64(value))))
}

func (s *StoragePackerV2) newBucket(key string, depth int32) *BucketV2 {
	bucket := &BucketV2{
		Key:     key,
		Buckets: make(map[string]*BucketV2),
		Items:   make(map[string]*Item),
		Depth:   depth,
	}

	return bucket
}

// NewStoragePackerV2 creates a new storage packer for a given view
func NewStoragePackerV2(config *Config) (*StoragePackerV2, error) {
	if config.View == nil {
		return nil, fmt.Errorf("nil view")
	}

	if config.ViewPrefix == "" {
		config.ViewPrefix = DefaultStoragePackerBucketsPrefix
	}

	if !strings.HasSuffix(config.ViewPrefix, "/") {
		config.ViewPrefix = config.ViewPrefix + "/"
	}

	if config.BucketBaseCount == 0 {
		config.BucketBaseCount = defaultBucketBaseCount
	}

	if config.BucketShardCount == 0 {
		config.BucketShardCount = defaultBucketShardCount
	}

	if config.BucketMaxSize == 0 {
		config.BucketMaxSize = defaultBucketMaxSize
	}

	// Create a new packer object for the given view
	packer := &StoragePackerV2{
		config:       config,
		bucketsCache: radix.New(),
	}

	return packer, nil
}
