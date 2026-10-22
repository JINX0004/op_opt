"""Static descriptor branches;windows/fill are proxies,not bank/cache claims."""
from collections import Counter
from functools import lru_cache
II_NAMES=('dense_calls','dense_mb','packed_odd_head','packed_odd_krows','reader_head','reader_first','reader_repeat','reader_krows','reader_first_krows','reader_repeat_krows')
EXTRA_NAMES=('dense_calls','dense_mb','packed_odd_calls','packed_odd_fill','reader_calls','reader_repeats_krows')
@lru_cache(maxsize=8192)
def work(rows,columns,pitch,phase=0):
 if min(rows,columns,pitch)<=0 or pitch<columns:raise ValueError('invalid static descriptor')
 visits=Counter();width=2*columns
 for r in range(rows):
  start=phase+2*r*pitch;visits.update(range(start//512,(start+width-1)//512+1))
 V=max(visits.values());packed_odd=pitch==columns and columns%32==16 and width<128;reader=pitch>columns
 first=int(V>1);repeat=max(0,V-2);krows=rows/1000;mb=rows*width/1e6
 dense=dict(dense_calls=1.,dense_mb=mb)
 ii=[dense,dict(packed_odd_head=float(packed_odd)),dict(packed_odd_krows=krows*packed_odd),dict(reader_head=float(reader),reader_first=first*reader,reader_repeat=repeat*reader),dict(reader_krows=krows*reader,reader_first_krows=krows*first*reader,reader_repeat_krows=krows*repeat*reader)]
 extra=[dense,dict(packed_odd_calls=float(packed_odd),packed_odd_fill=min(1,rows*width/8192)*packed_odd),dict(reader_calls=float(reader),reader_repeats_krows=krows*max(0,V-1)*reader)]
 return dict(interval=ii,extra_completion=extra,source_window_max_visits=V,packed_odd=packed_odd,strided=reader)