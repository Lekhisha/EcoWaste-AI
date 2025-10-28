
import img1 from "../../assests/image1.jpg";

const bgstyle2 = {
  backgroundImage: `url(${img1})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const Uses = () => {
  return (
    <>
     <div className="relative py-14">
        <div className="container py-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 place-items-center">
            <div className="rounded-3xl overflow-hidden  shadow-2xl" style={bgstyle2}>
              <img src={img1} alt="" className="w-full" />
            </div>

            <div>
              <div className="text-3xl font-semibold text-cyan-950">
                Sustainability Benefits
              </div>
              <p className="py-4 font-semibold text-cyan-950">
                By using SmartWaste Guide, you're contributing to a more
                sustainable future. Proper waste sorting leads to higher
                recycling rates, less landfill waste, and a reduced carbon
                footprint. Join us in making a tangible positive impact on our
                environment.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default Uses
