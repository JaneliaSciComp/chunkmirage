// Fiji follows a chunkmirage server: its datasets in one BigDataViewer, their chunks fetched
// again on every edit of a pipeline (a slider on the control page, new code, a PUT), so
// BigDataViewer shows what the python viewer shows, keeping its view and brightness settings.
//
// In Fiji: press [ for the script editor, pick Language > Groovy in the menu bar (or save
// this as a .groovy file, which picks it), Run. A dialog asks for the two things that change
// from one server to the next, and nothing else ever does:
//   - the control UI link the server printed, e.g. http://localhost:8004/ui?token=...: the
//     port it chose and its token (a server restarted has a new token, so a new link);
//   - the datasets to open, comma separated, one BigDataViewer source each: raw,processed
//     for a plain `chunkmirage serve`; the --name given, in place of processed; no raw
//     with --no-raw; every name wanted when a server has more (a Python registry).
// Nothing here is about the pipeline: the same script follows any server, any ops, any
// Python, since the server does the work and this only fetches again. A BigDataViewer
// opens with the sources; press S for their ranges (labels: 0 to about their count).
// Close it to stop following.
//
// How: BigDataViewer keeps each dataset's chunks in a cache in memory, and its addresses carry
// no digest, so emptying those caches and repainting fetches the current pipeline.

#@ String (label = "chunkmirage link (the control UI link the server printed)", value = "http://localhost:8000/ui") link
#@ String (label = "datasets, comma separated", value = "raw,processed") names

import bdv.util.BdvOptions
import net.imglib2.cache.img.CachedCellImg
import org.janelia.saalfeldlab.n5.bdv.N5Viewer

import javax.swing.SwingUtilities
import javax.swing.Timer
import java.awt.event.ActionListener
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent

def url = new URL(link.trim())
def server = "${url.protocol}://${url.authority}".toString()
def token = (url.query ?: "").split("&").findResult { it.startsWith("token=") ? it.substring(6) : null }

// The event stream first, so a wrong link or token says so before any window opens
def events = new URL("${server}/api/events" + (token ? "?token=${token}" : "")).openConnection()
events.setRequestProperty("Accept", "text/event-stream")
if (events.responseCode != 200)
    throw new RuntimeException("${server}/api/events answered ${events.responseCode}: paste the control UI link the server printed, with its ?token=")
def stream = events.inputStream

def uris = names.split(",").collect { "${server}/${it.trim()}/zarr".toString() } as String[]
def bdv = N5Viewer.show(uris, BdvOptions.options().frameTitle("chunkmirage: ${names}"))
def panel = bdv.getViewerPanel()

// The cached image under the views a source wraps around it (slices, added axes)
def cached = { img ->
    for (int i = 0; img != null && i < 20; i++) {
        if (img instanceof CachedCellImg) return img
        try { img = img.getSource() } catch (MissingMethodException e) { return null }
    }
    null
}

def refetch = {
    int emptied = 0
    def state = panel.state()
    state.getSources().each { sac ->
        def src = sac.getSpimSource()
        for (int level = 0; level < src.getNumMipmapLevels(); level++) {
            def img = cached(src.getSource(state.getCurrentTimepoint(), level))
            if (img != null) { img.getCache().invalidateAll(); emptied++ }
        }
    }
    panel.requestRepaint()
    println "chunkmirage: ${emptied} caches emptied, BigDataViewer refetching"
}

def timer = new Timer(300, { refetch() } as ActionListener)  // a slider drag is many edits
timer.repeats = false

SwingUtilities.getWindowAncestor(panel).addWindowListener(new WindowAdapter() {
    void windowClosed(WindowEvent e) { stream.close() }
})
def follower = new Thread({
    def reader = new BufferedReader(new InputStreamReader(stream, "UTF-8"))
    boolean first = true  // the stream opens with the pipelines as they are
    try {
        for (def line = reader.readLine(); line != null; line = reader.readLine()) {
            if (!line.startsWith("data:")) continue
            if (first) { first = false; continue }
            timer.restart()
        }
    } catch (IOException ignored) {}  // the window closed, or the server went away
    println "chunkmirage: no longer following ${server}"
} as Runnable)
follower.daemon = true
follower.start()
println "chunkmirage: following ${server}; edit a pipeline and BigDataViewer refetches"
